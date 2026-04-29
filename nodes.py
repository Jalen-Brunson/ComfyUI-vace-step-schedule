import logging

import torch
import torch.nn.functional as F

import comfy.conds
import comfy.hooks
import comfy.latent_formats
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import folder_paths
import latent_preview
import node_helpers
import nodes

log = logging.getLogger(__name__)


def parse_float_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        cleaned = value.replace("[", "").replace("]", "").replace(";", ",").replace("\n", ",")
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        return [float(p) for p in parts]
    raise TypeError(f"Cannot parse float list from {type(value).__name__}")


def _max_existing_vace_T(cond_list):
    """Return the largest temporal dim across any already-stamped vace_frames in a conditioning."""
    max_t = 0
    if not cond_list:
        return max_t
    for entry in cond_list:
        if isinstance(entry, (list, tuple)) and len(entry) > 1 and isinstance(entry[1], dict):
            for vf in entry[1].get("vace_frames", []):
                if hasattr(vf, "shape") and len(vf.shape) >= 3 and vf.shape[2] > max_t:
                    max_t = vf.shape[2]
    return max_t


def _neutral_vace_pad(ref_tensor, pad_size):
    """Build a pad tensor that lands as 0 in VACE view after native's process_latent_in.

    Native's WAN21.extra_conds always applies `process_latent_in` = (x - mean) / std to the
    VACE context. To have the VACE block see 0 at padded positions, the stamped value must
    be `latents_mean`, since process_in(mean) = 0. We achieve that by using process_out(zeros)
    per 16-channel slab.

    Mask channels are exempt — they're not part of the VAE-normalized content and don't go
    through process_latent_in.
    """
    zeros = torch.zeros_like(ref_tensor[:, :, :pad_size])
    lf = comfy.latent_formats.Wan21()
    if zeros.shape[1] % 16 == 0 and zeros.shape[1] <= 32:
        for i in range(0, zeros.shape[1], 16):
            zeros[:, i:i + 16] = lf.process_out(zeros[:, i:i + 16])
    return zeros


def _pad_vace_in_cond(cond_list, pad_size):
    """Prepend `pad_size` neutral frames to every vace_frames / vace_mask entry on every cond.

    Non-ref encoders chained ahead of a ref encoder must be padded at the front so their T
    matches the ref encoder's (required by torch.stack in model_base.WAN21.extra_conds).
    That padding must look like "no contribution" at the VACE block — i.e. zero. Since
    native always runs process_latent_in on vace_frames, we stamp latents_mean (which
    normalizes to 0) rather than raw zeros (which normalize to -mean/std ≈ ±0.5 garbage).
    Mask pads are kept as raw zeros since masks don't get normalized.
    """
    if pad_size <= 0 or not cond_list:
        return cond_list
    new_cond = []
    for entry in cond_list:
        if not (isinstance(entry, (list, tuple)) and len(entry) > 1 and isinstance(entry[1], dict)):
            new_cond.append(entry)
            continue
        meta = dict(entry[1])
        frames = meta.get("vace_frames")
        if frames:
            new_frames = []
            for vf in frames:
                pad = _neutral_vace_pad(vf, pad_size)
                new_frames.append(torch.cat([pad, vf], dim=2))
            meta["vace_frames"] = new_frames
        masks = meta.get("vace_mask")
        if masks:
            new_masks = []
            for vm in masks:
                pad = torch.zeros_like(vm[:, :, :pad_size])
                new_masks.append(torch.cat([pad, vm], dim=2))
            meta["vace_mask"] = new_masks
        rebuilt = list(entry)
        rebuilt[1] = meta
        new_cond.append(type(entry)(rebuilt) if isinstance(entry, tuple) else rebuilt)
    return new_cond


def _pad_to_target_aspect(image_BHWC, target_width, target_height, pad_value=1.0):
    """Pad image with `pad_value` to match target W/H aspect ratio without cropping.

    Input: (B, H, W, C) in [0, 1]. Output: (B, H', W', C) with new H'/W' matching target aspect.
    Matches WanVideoWrapper's ref-image padding (nodes.py:1773-1789).
    """
    B, H, W, C = image_BHWC.shape
    current_aspect = W / H
    target_aspect = target_width / target_height
    if current_aspect > target_aspect:
        new_h = int(W / target_aspect)
        pad_h = (new_h - H) // 2
        padded = torch.full((B, new_h, W, C), pad_value, device=image_BHWC.device, dtype=image_BHWC.dtype)
        padded[:, pad_h:pad_h + H, :, :] = image_BHWC
        return padded
    elif current_aspect < target_aspect:
        new_w = int(H * target_aspect)
        pad_w = (new_w - W) // 2
        padded = torch.full((B, H, new_w, C), pad_value, device=image_BHWC.device, dtype=image_BHWC.dtype)
        padded[:, :, pad_w:pad_w + W, :] = image_BHWC
        return padded
    return image_BHWC


def _resolve_schedule(string_val, socket_val):
    """Prefer a connected list-producing socket over the string widget.

    `socket_val` from a FLOAT input that's still showing its widget arrives as a
    scalar (e.g. 1.0). When the widget is converted to an input and connected to
    a node like KJNodes' `StringToFloatList`, it arrives as a Python list. We
    only honor the socket when it's actually a list — otherwise the string wins.
    """
    if isinstance(socket_val, (list, tuple)) and len(socket_val) > 0:
        return [float(v) for v in socket_val]
    return parse_float_list(string_val) if string_val else []


def _step_idx_from_sigma(sigma_value, sigmas_cpu):
    if sigmas_cpu.numel() == 0:
        return 0
    diffs = (sigmas_cpu - float(sigma_value)).abs()
    return int(diffs.argmin().item())


def _value_for_step(schedule, step_idx, total_steps):
    n = len(schedule)
    if n == 0:
        return None
    if n == 1:
        return float(schedule[0])
    if n == total_steps:
        if step_idx < 0:
            return float(schedule[0])
        if step_idx >= n:
            return float(schedule[-1])
        return float(schedule[step_idx])
    if total_steps <= 1:
        return float(schedule[0])
    t = max(0.0, min(1.0, step_idx / max(1, total_steps - 1)))
    idx = int(round(t * (n - 1)))
    return float(schedule[idx])


def _compute_sigmas_for_run(patcher, sampler_name, scheduler, steps, denoise):
    ks = comfy.samplers.KSampler(
        patcher,
        steps=steps,
        device=patcher.load_device,
        sampler=sampler_name,
        scheduler=scheduler,
        denoise=denoise,
        model_options=patcher.model_options,
    )
    return ks.sigmas.detach().cpu().clone()


def _make_patched_extra_conds(original_extra_conds_bound):
    def patched(**kwargs):
        out = original_extra_conds_bound(**kwargs)
        sched = kwargs.get("vace_strength_schedule", None)
        if sched is not None:
            out["vace_strength_schedule"] = comfy.conds.CONDConstant(sched)
        ref_sched = kwargs.get("vace_ref_strength_schedule", None)
        if ref_sched is not None:
            out["vace_ref_strength_schedule"] = comfy.conds.CONDConstant(ref_sched)
        num_ref_slots = kwargs.get("vace_num_ref_slots", None)
        if num_ref_slots is not None:
            out["vace_num_ref_slots"] = comfy.conds.CONDConstant(num_ref_slots)
        return out
    return patched


def _make_unet_wrapper(sigmas_cpu, total_steps, verbose=False):
    state = {"last_logged_step": -1}

    def wrapper(apply_model, args):
        input_x = args["input"]
        timestep = args["timestep"]
        c = args["c"]

        schedules = c.get("vace_strength_schedule", None)
        if schedules is None:
            return apply_model(input_x, timestep, **c)

        sigma_val = float(timestep.flatten()[0].detach().cpu().item())
        step_idx = _step_idx_from_sigma(sigma_val, sigmas_cpu)

        ref_schedules = c.get("vace_ref_strength_schedule", None) or [None] * len(schedules)
        num_ref_slots_list = c.get("vace_num_ref_slots", None) or [0] * len(schedules)

        # Derive spatial/temporal dims from vace_context to build per-position strength tensors
        # when ref_strength != content_strength. Shape: (B, num_encoders, 96, T, H_lat, W_lat).
        vace_ctx = c.get("vace_context", None)
        tokens_per_T, total_T = None, None
        if vace_ctx is not None and hasattr(vace_ctx, "shape") and vace_ctx.ndim >= 6:
            _, _, _, T_lat, H_lat, W_lat = vace_ctx.shape
            tokens_per_T = (H_lat // 2) * (W_lat // 2)
            total_T = T_lat

        existing = c.get("vace_strength", None) or [1.0] * len(schedules)
        new_strengths = list(existing)
        content_log, ref_log = [], []

        for i, sched in enumerate(schedules):
            if sched is None:
                content_log.append(None)
                ref_log.append(None)
                continue
            content_str = _value_for_step(sched, step_idx, total_steps)
            if content_str is None:
                content_log.append(None)
                ref_log.append(None)
                continue

            ref_sched_i = ref_schedules[i] if i < len(ref_schedules) else None
            num_refs = int(num_ref_slots_list[i]) if i < len(num_ref_slots_list) else 0
            ref_str = None
            if ref_sched_i:
                ref_str = _value_for_step(ref_sched_i, step_idx, total_steps)

            use_tensor = (
                ref_str is not None
                and ref_str != content_str
                and num_refs > 0
                and tokens_per_T is not None
                and total_T is not None
            )

            if use_tensor:
                # Build (1, seq_len, 1) that broadcasts against c_skip of shape (1, seq_len, dim).
                # First `num_refs * tokens_per_T` positions get ref_str, rest get content_str.
                seq_len = total_T * tokens_per_T
                ref_tokens = num_refs * tokens_per_T
                strength_t = torch.full(
                    (1, seq_len, 1), float(content_str),
                    dtype=input_x.dtype, device=input_x.device,
                )
                strength_t[:, :ref_tokens, :] = float(ref_str)
                val = strength_t
            else:
                val = float(content_str)

            if i < len(new_strengths):
                new_strengths[i] = val
            else:
                new_strengths.append(val)

            content_log.append(content_str)
            ref_log.append(ref_str if ref_str is not None else content_str)

        c = dict(c)
        c["vace_strength"] = new_strengths
        c.pop("vace_strength_schedule", None)
        c.pop("vace_ref_strength_schedule", None)
        c.pop("vace_num_ref_slots", None)

        if verbose and step_idx != state["last_logged_step"]:
            state["last_logged_step"] = step_idx
            def _fmt(xs):
                return ", ".join(f"{v:.3f}" if v is not None else "-" for v in xs)
            if any(r is not None and c_ is not None and r != c_ for r, c_ in zip(ref_log, content_log)):
                log.info(f"[VaceSched] step={step_idx} sigma={sigma_val:.4f} content=[{_fmt(content_log)}] ref=[{_fmt(ref_log)}]")
            else:
                log.info(f"[VaceSched] step={step_idx} sigma={sigma_val:.4f} vace_strength=[{_fmt(content_log)}]")

        return apply_model(input_x, timestep, **c)
    return wrapper


def _make_cfg_function(cfg_schedule, sigmas_cpu, total_steps, verbose=False):
    state = {"last_logged_step": -1}

    def cfg_fn(args):
        cond_noise = args["cond"]
        uncond_noise = args["uncond"]
        fallback = args["cond_scale"]
        sigma = args["sigma"]

        if hasattr(sigma, "flatten"):
            sigma_val = float(sigma.flatten()[0].detach().cpu().item())
        else:
            sigma_val = float(sigma)
        step_idx = _step_idx_from_sigma(sigma_val, sigmas_cpu)
        scale = _value_for_step(cfg_schedule, step_idx, total_steps)
        if scale is None:
            scale = float(fallback)

        if verbose and step_idx != state["last_logged_step"]:
            state["last_logged_step"] = step_idx
            log.info(f"[VaceSched] step={step_idx} sigma={sigma_val:.4f} cfg={scale:.3f}")

        return uncond_noise - (uncond_noise - cond_noise) * float(scale)
    return cfg_fn


def _make_lora_probe_post_cfg(sigmas_cpu, total_steps):
    """post_cfg callback that logs active LoRA hook keyframe strengths once per step."""
    state = {"last_logged_step": -1}

    def probe(args):
        denoised = args.get("denoised")
        sigma = args.get("sigma")
        model = args.get("model")
        patcher = getattr(model, "current_patcher", None)
        if patcher is None or getattr(patcher, "current_hooks", None) is None:
            return denoised
        hooks = patcher.current_hooks.hooks
        if not hooks:
            return denoised

        sigma_val = (float(sigma.flatten()[0].detach().cpu().item())
                     if hasattr(sigma, "flatten") else float(sigma))
        step_idx = _step_idx_from_sigma(sigma_val, sigmas_cpu)
        if step_idx == state["last_logged_step"]:
            return denoised
        state["last_logged_step"] = step_idx

        parts = []
        for h in hooks:
            kf_group = getattr(h, "hook_keyframe", None)
            if kf_group is None:
                continue
            cur = getattr(kf_group, "_current_keyframe", None)
            if cur is None:
                continue
            parts.append(f"{cur.strength:.3f}@{cur.start_percent:.3f}")
        if parts:
            log.info(f"[VaceSched] step={step_idx} lora_hooks=[{', '.join(parts)}]")
        return denoised

    return probe


class WanVaceToVideoScheduled:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "strength_schedule": ("STRING", {
                    "default": "1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
                    "multiline": False,
                    "tooltip": "Per-step strength applied at CONTENT positions (non-ref temporal slots). Comma-separated floats. One value = flat. Ignored if strength_list socket is connected.",
                }),
            },
            "optional": {
                "control_video": ("IMAGE",),
                "control_masks": ("MASK",),
                "reference_image": ("IMAGE",),
                "strength_list": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "Optional FLOAT list for strength_schedule (convert widget to input, connect KJNodes 'String to Float List'). Overrides the string.",
                }),
                "ref_strength_schedule": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Per-step strength applied ONLY at the reference-image latent slot(s). Comma-separated floats, one per step.\n"
                        "Leave empty (default) to use strength_schedule for ref positions too — equivalent to the previous single-strength behavior.\n"
                        "Use this when you want ref anchor at lower strength than control signal, e.g. ref_strength_schedule='0.1, 0.1, 0, 0, 0, 0' paired with strength_schedule='0.8, 0.8, 0.8, 0.8, 0.8, 0.8' — weak ref anchor, strong control. Fixes 'tattoo marks' artifact when ref contribution is strong at high-noise steps.\n"
                        "Only meaningful when reference_image is connected; ignored otherwise."
                    ),
                }),
                "ref_strength_list": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "Optional FLOAT list for ref_strength_schedule (convert widget to input, connect KJNodes 'String to Float List'). Overrides the string.",
                }),
                "ref_resize_mode": (["lanczos_pad", "lanczos_crop", "bilinear_crop"], {
                    "default": "lanczos_pad",
                    "tooltip": (
                        "How to resize the reference image(s) to target dimensions.\n"
                        "- lanczos_pad (default, matches WanVideoWrapper): pad with white to target aspect, then lanczos resize. Preserves full ref content, sharper latent.\n"
                        "- lanczos_crop: center-crop to target aspect, lanczos resize. Sharper but may cut off subject edges.\n"
                        "- bilinear_crop (stock native behavior): center-crop + bilinear. Matches stock WanVaceToVideo exactly. Softer latent."
                    ),
                }),
                "multi_ref_mode": (["horizontal_tile", "separate_frames"], {
                    "default": "horizontal_tile",
                    "tooltip": (
                        "How to handle a ref_image batch of N>1 images.\n"
                        "- horizontal_tile (default, matches WanVideoWrapper nodes.py:1770): concatenate all refs side-by-side along width into one wide image, then aspect-pad + resize + encode as a SINGLE latent slot. All refs visible in the same ref frame; VACE attention treats them as a single multi-region anchor. Typical for multi-angle identity refs.\n"
                        "- separate_frames: encode each ref as its own latent temporal slot (N slots total, trim_latent=N). VACE sees them as a time-progression prefix rather than parallel anchors; usually only the first ref dominates and later ones get overwritten in the attention. NOT what VACE was trained on for multi-ref. Kept for debugging only."
                    ),
                }),
                "ref_reactive_mode": (["process_out_zeros", "zeros"], {
                    "default": "process_out_zeros",
                    "tooltip": (
                        "Fill value for the reactive channel at ref-frame positions.\n"
                        "- process_out_zeros (default, matches WanVideoWrapper): stamp latents_mean. After native's process_latent_in it becomes exactly 0 — same as what wrapper's VACE block sees (wrapper's VAE normalizes internally, so its torch.zeros_like already IS zero).\n"
                        "- zeros: stamp raw zeros. Native's process_latent_in turns them into -mean/std (non-zero per-channel constants around ±0.5), which VACE wasn't trained on. Causes facial blotches and other reactive-channel artifacts. DEBUG ONLY."
                    ),
                }),
                "vace_latent_space": (["normalized", "raw"], {
                    "default": "normalized",
                    "tooltip": (
                        "Controls whether native ComfyUI's `extra_conds` normalization (process_latent_in on VACE frames) is cancelled out.\n"
                        "- normalized (default, correct): let native's process_latent_in pass through. VACE block sees normalized latents — equivalent to WanVideoWrapper's behavior, since the wrapper's VAE encode applies normalization internally (vae.py:536). Both paths converge at the VACE block with normalized latents.\n"
                        "- raw (debug only, usually WRONG): pre-apply process_out so native's process_in cancels out. VACE block sees raw VAE output — NOT what the model expects. Produces severe artifacts. Kept as a toggle for A/B comparison."
                    ),
                }),
                "control_resize_mode": (["lanczos_nocrop", "bilinear_centercrop"], {
                    "default": "lanczos_nocrop",
                    "tooltip": (
                        "How control_video is resized to target dimensions.\n"
                        "- lanczos_nocrop (default, matches WanVideoWrapper): preserves thin pose/depth lines, no cropping. Expects control_video aspect to match target (pre-resize upstream).\n"
                        "- bilinear_centercrop (stock WanVaceToVideo): bilinear interpolation blurs thin lines, center-crops mismatched aspects. Older default; usually produces less precise motion guidance."
                    ),
                }),
                "mask_resize_mode": (["nearest_nocrop", "bilinear_centercrop"], {
                    "default": "nearest_nocrop",
                    "tooltip": (
                        "How control_masks is resized to target dimensions.\n"
                        "- nearest_nocrop (default, matches WanVideoWrapper): keeps hard mask edges crisp (binary 0/1 preserved). No cropping.\n"
                        "- bilinear_centercrop (stock WanVaceToVideo): bilinear on mask produces soft edge gradient boundaries, causing 'mask outline visible in first frame' artifacts. Kept for compatibility only."
                    ),
                }),
                "inactive_fill_mode": (["black_0.0", "gray_0.5"], {
                    "default": "black_0.0",
                    "tooltip": (
                        "Pixel value used in the empty side of the inactive/reactive split before VAE encode.\n"
                        "- black_0.0 (default, matches WanVideoWrapper nodes.py:1831): inactive=input*(1-mask), reactive=input*mask. Empty regions encode to the 'absent' latent pattern VACE was trained on. Large masks leave no stain.\n"
                        "- gray_0.5 (stock WanVaceToVideo nodes_wan.py:339): inactive=(input-0.5)*(1-mask)+0.5, reactive=(input-0.5)*mask+0.5. Empty regions become mid-gray, which encodes to a nonzero latent pattern VACE tries to reconstruct, producing visible gray stains/blinking when the mask is wider than the subject. Kept for stock-ComfyUI A/B compat."
                    ),
                }),
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Log the resolved strength schedule to the ComfyUI console at execute time.",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_latent")
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"

    def execute(self, positive, negative, vae, width, height, length, batch_size, strength_schedule,
                control_video=None, control_masks=None, reference_image=None, strength_list=None,
                ref_strength_schedule="", ref_strength_list=None,
                ref_resize_mode="lanczos_pad", ref_reactive_mode="process_out_zeros",
                multi_ref_mode="horizontal_tile", vace_latent_space="normalized",
                control_resize_mode="lanczos_nocrop", mask_resize_mode="nearest_nocrop",
                inactive_fill_mode="black_0.0",
                verbose=False):
        schedule = _resolve_schedule(strength_schedule, strength_list)
        # Only use ref_strength_schedule when the socket is connected OR the string is non-empty;
        # otherwise we pass None so the unet wrapper falls back to single-strength behavior.
        ref_sched_connected = isinstance(ref_strength_list, (list, tuple)) and len(ref_strength_list) > 0
        ref_sched_string_set = bool(ref_strength_schedule and ref_strength_schedule.strip())
        ref_schedule = _resolve_schedule(ref_strength_schedule, ref_strength_list) if (ref_sched_connected or ref_sched_string_set) else None
        if verbose:
            log.info(f"[WanVaceToVideoScheduled] schedule={schedule} fallback_strength={schedule[0] if schedule else 1.0} "
                     f"has_ref_image={reference_image is not None} ref_schedule={ref_schedule} "
                     f"ref_resize_mode={ref_resize_mode} ref_reactive_mode={ref_reactive_mode} "
                     f"multi_ref_mode={multi_ref_mode} vace_latent_space={vace_latent_space} "
                     f"control_resize_mode={control_resize_mode} mask_resize_mode={mask_resize_mode} "
                     f"inactive_fill_mode={inactive_fill_mode}")
        fallback = float(schedule[0]) if schedule else 1.0

        # Snap dimensions to a multiple of (VAE_STRIDE * spatial_patch_size) = 8 * 2 = 16.
        # Two constraints stack:
        #   (1) mask.view(length, H//8, 8, W//8, 8) needs H,W divisible by 8 (the VAE stride).
        #   (2) WAN's _forward calls pad_to_patch_size(x) with patch_size=(1,2,2), so the
        #       sampler latent's H/8,W/8 get padded UP to the next even integer. But
        #       vace_patch_embedding receives vace_context UNPADDED. If H/8 is odd, x gets
        #       padded to H/8+1 while c stays at H/8 — producing mismatched seq_len at
        #       comfy/ldm/wan/model.py:285 (`c = self.before_proj(c) + x`).
        # Snapping pixel dims to a multiple of 16 makes H/8 always even, so no asymmetric
        # padding occurs. Use ceil so we don't crop user content (4-pixel pad is invisible).
        VAE_STRIDE = 8
        PATCH_SPATIAL = 2  # WAN patch_size[1] == patch_size[2]
        ALIGN = VAE_STRIDE * PATCH_SPATIAL  # = 16
        orig_w, orig_h = width, height
        width = ((width + ALIGN - 1) // ALIGN) * ALIGN
        height = ((height + ALIGN - 1) // ALIGN) * ALIGN
        if (orig_w, orig_h) != (width, height):
            log.warning(
                f"[WanVaceToVideoScheduled] dimensions snapped UP to multiples of {ALIGN}: "
                f"({orig_w}x{orig_h}) -> ({width}x{height})"
            )

        # Resize mode dispatch for control_video and mask. Default mirrors WanVideoWrapper.
        if control_resize_mode == "lanczos_nocrop":
            ctrl_interp, ctrl_crop = "lanczos", "disabled"
        else:
            ctrl_interp, ctrl_crop = "bilinear", "center"
        if mask_resize_mode == "nearest_nocrop":
            mask_interp, mask_crop = "nearest-exact", "disabled"
        else:
            mask_interp, mask_crop = "bilinear", "center"

        latent_length = ((length - 1) // 4) + 1
        if control_video is not None:
            control_video = comfy.utils.common_upscale(
                control_video[:length].movedim(-1, 1), width, height, ctrl_interp, ctrl_crop
            ).movedim(1, -1)
            if control_video.shape[0] < length:
                control_video = F.pad(
                    control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5
                )
        else:
            control_video = torch.ones((length, height, width, 3)) * 0.5

        if reference_image is not None:
            num_refs_input = int(reference_image.shape[0])
            ref_in = reference_image[..., :3]

            # Multi-ref handling: tile horizontally into one wide image (matches wrapper),
            # or keep separate frames (old behavior, usually wrong for VACE attention).
            if num_refs_input > 1 and multi_ref_mode == "horizontal_tile":
                # ref_in shape (N, H, W, 3). Cat along dim=2 (width) after stripping batch:
                ref_in = torch.cat([ref_in[i] for i in range(num_refs_input)], dim=1).unsqueeze(0)
                # Now shape (1, H, W * N, 3) — one wide image containing all refs side-by-side.

            if ref_resize_mode == "lanczos_pad":
                ref_in = _pad_to_target_aspect(ref_in, width, height, pad_value=1.0)
                interp, crop = "lanczos", "disabled"
            elif ref_resize_mode == "lanczos_crop":
                interp, crop = "lanczos", "center"
            else:
                interp, crop = "bilinear", "center"

            ref_in = comfy.utils.common_upscale(
                ref_in.movedim(-1, 1), width, height, interp, crop
            ).movedim(1, -1)

            # Encode. One VAE call per batch item; shape[0] is 1 (tiled) or N (separate).
            encoded = []
            for i in range(ref_in.shape[0]):
                encoded.append(vae.encode(ref_in[i:i + 1, :, :, :3]))
            reference_image = torch.cat(encoded, dim=2)
            num_ref_slots = reference_image.shape[2]

            if ref_reactive_mode == "zeros":
                reactive_fill = torch.zeros_like(reference_image)
            else:
                reactive_fill = comfy.latent_formats.Wan21().process_out(torch.zeros_like(reference_image))
            reference_image = torch.cat([reference_image, reactive_fill], dim=1)

            if verbose:
                log.info(f"[WanVaceToVideoScheduled] ref_image batch={num_refs_input} "
                         f"multi_ref_mode={multi_ref_mode} -> {num_ref_slots} latent ref slot(s) "
                         f"(resize={interp}+{crop}, reactive={ref_reactive_mode})")

        if control_masks is None:
            mask = torch.ones((length, height, width, 1))
        else:
            mask = control_masks
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = comfy.utils.common_upscale(mask[:length], width, height, mask_interp, mask_crop).movedim(1, -1)
            if mask.shape[0] < length:
                mask = F.pad(mask, (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]), value=1.0)

        if inactive_fill_mode == "black_0.0":
            inactive = control_video * (1 - mask)
            reactive = control_video * mask
        else:
            cv_centered = control_video - 0.5
            inactive = (cv_centered * (1 - mask)) + 0.5
            reactive = (cv_centered * mask) + 0.5

        inactive = vae.encode(inactive[:, :, :, :3])
        reactive = vae.encode(reactive[:, :, :, :3])
        control_video_latent = torch.cat((inactive, reactive), dim=1)
        if reference_image is not None:
            control_video_latent = torch.cat((reference_image, control_video_latent), dim=2)

        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        mask = mask.permute(2, 4, 0, 1, 3)
        mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
        mask = F.interpolate(
            mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode="nearest-exact"
        ).squeeze(0)

        trim_latent = 0
        if reference_image is not None:
            mask_pad = torch.zeros_like(mask[:, :reference_image.shape[2], :, :])
            mask = torch.cat((mask_pad, mask), dim=1)
            latent_length += reference_image.shape[2]
            trim_latent = reference_image.shape[2]

        mask = mask.unsqueeze(0)

        existing_T = max(_max_existing_vace_T(positive), _max_existing_vace_T(negative))
        current_T = control_video_latent.shape[2]
        target_T = max(existing_T, current_T)

        if current_T < target_T:
            pad = target_T - current_T
            pad_cvl = torch.zeros_like(control_video_latent[:, :, :pad])
            # We want VACE to see 0 at padded positions after native's process_latent_in.
            # In 'normalized' mode there's no pre-transform later, so the stamped value
            # must already equal latents_mean to normalize to zero.
            # In 'raw' mode the pre-transform loop below runs process_out on this pad too,
            # which converts raw zeros into latents_mean (process_out(0) = mean) — same
            # end result after native's normalization. So we only need to mean-pad here
            # when we're NOT going to pre-transform afterward.
            if vace_latent_space == "normalized":
                _lf = comfy.latent_formats.Wan21()
                if pad_cvl.shape[1] % 16 == 0 and pad_cvl.shape[1] <= 32:
                    for i in range(0, pad_cvl.shape[1], 16):
                        pad_cvl[:, i:i + 16] = _lf.process_out(pad_cvl[:, i:i + 16])
            control_video_latent = torch.cat([pad_cvl, control_video_latent], dim=2)
            pad_mask = torch.zeros_like(mask[:, :, :pad])
            mask = torch.cat([pad_mask, mask], dim=2)
            trim_latent = max(trim_latent, pad)
            latent_length = target_T
        elif existing_T > 0 and current_T > existing_T:
            pad = current_T - existing_T
            positive = _pad_vace_in_cond(positive, pad)
            negative = _pad_vace_in_cond(negative, pad)

        if vace_latent_space == "raw":
            # Pre-apply the inverse of `process_latent_in`. Native's WAN21.extra_conds
            # (model_base.py:1380-1381) runs `process_latent_in` on every 16-channel slab of
            # the VACE context, which normalizes it into zero-mean unit-std space. The VACE
            # weights were trained against raw VAE latents (WanVideoWrapper's forward_vace
            # never normalizes), so native's extra normalization shifts the input distribution
            # and produces skin/texture artifacts at ref_strength > 0.4.
            # `process_in(process_out(x)) == x`, so pre-applying `process_out` here makes the
            # VACE block see the raw VAE latent space — matching the wrapper. This runs
            # AFTER the local padding so zero-padded positions also get pre-transformed,
            # becoming latents_mean → normalized to 0 by native — clean zero in VACE view.
            lf = comfy.latent_formats.Wan21()
            for i in range(0, control_video_latent.shape[1], 16):
                control_video_latent[:, i:i + 16] = lf.process_out(control_video_latent[:, i:i + 16])

        # num_ref_slots for this encoder = ref temporal slots at the FRONT of control_video_latent
        # after all prepending. For ref-on encoder: = reference_image.shape[2] before padding logic
        # (which is 1 for horizontal_tile or N for separate_frames). For non-ref encoder: 0.
        # Padded slots from downstream retroactive pad are NOT counted as "ref" for this encoder —
        # they're neutral mean-valued and contribute 0 to VACE regardless of strength tensor.
        if reference_image is not None:
            num_ref_slots_this_encoder = int(reference_image.shape[2])
        else:
            num_ref_slots_this_encoder = 0

        vals = {
            "vace_frames": [control_video_latent],
            "vace_mask": [mask],
            "vace_strength": [fallback],
            "vace_strength_schedule": [schedule],
            "vace_trim_latent": [int(trim_latent)],
            "vace_ref_strength_schedule": [ref_schedule],      # list-or-None per encoder
            "vace_num_ref_slots": [num_ref_slots_this_encoder],
        }
        positive = node_helpers.conditioning_set_values(positive, vals, append=True)
        negative = node_helpers.conditioning_set_values(negative, vals, append=True)

        latent = torch.zeros(
            [batch_size, 16, latent_length, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        return (positive, negative, {"samples": latent}, trim_latent)


class VaceScheduleKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": ("BOOLEAN", {"default": True}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01,
                                   "tooltip": "Fallback CFG used if cfg_schedule is blank."}),
                "cfg_schedule": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Per-step CFG, comma-separated. Blank = flat 'cfg'. Ignored if cfg_list socket is connected.",
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": ("BOOLEAN", {"default": False}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "cfg_list": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Right-click this widget → 'Convert Widget to Input', then connect a FLOAT-list node (KJNodes' 'String to Float List', etc.). The list overrides cfg_schedule. The scalar widget value is ignored.",
                }),
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Log per-step sigma, resolved CFG, resolved VACE strengths, and active LoRA keyframe strengths to the ComfyUI console. One line per unique step.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, model, add_noise, noise_seed, steps, cfg, cfg_schedule, sampler_name, scheduler,
               positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise,
               cfg_list=None, verbose=False):
        cfg_list = _resolve_schedule(cfg_schedule, cfg_list)
        if verbose:
            log.info(f"[VaceScheduleKSampler] sampler={sampler_name} scheduler={scheduler} steps={steps} "
                     f"denoise={denoise} start={start_at_step} end={end_at_step} cfg_schedule={cfg_list or 'flat=' + str(cfg)}")

        m = model.clone()

        sigmas_cpu = _compute_sigmas_for_run(m, sampler_name, scheduler, steps, denoise)
        if verbose:
            log.info(f"[VaceScheduleKSampler] sigmas={[f'{float(s):.4f}' for s in sigmas_cpu.tolist()]}")

        original_extra_conds = m.model.extra_conds
        m.add_object_patch("extra_conds", _make_patched_extra_conds(original_extra_conds))

        m.set_model_unet_function_wrapper(_make_unet_wrapper(sigmas_cpu, steps, verbose=verbose))

        if cfg_list:
            # Only force uncond computation if any scheduled cfg value > 1.
            # When every scheduled value is <= 1 there's no guidance being applied, so skipping
            # uncond matches stock behavior and halves forward-pass VRAM at those steps.
            needs_uncond = any(v > 1.0 for v in cfg_list)
            m.set_model_sampler_cfg_function(
                _make_cfg_function(cfg_list, sigmas_cpu, steps, verbose=verbose),
                disable_cfg1_optimization=needs_uncond,
            )
            if verbose:
                log.info(f"[VaceScheduleKSampler] disable_cfg1_optimization={needs_uncond} (any(cfg>1)={needs_uncond})")

        if verbose:
            m.set_model_sampler_post_cfg_function(_make_lora_probe_post_cfg(sigmas_cpu, steps))

        latent = latent_image
        latent_samples = latent["samples"]
        latent_samples = comfy.sample.fix_empty_latent_channels(m, latent_samples)

        if add_noise:
            batch_inds = latent.get("batch_index", None)
            noise = comfy.sample.prepare_noise(latent_samples, noise_seed, batch_inds)
        else:
            noise = torch.zeros(
                latent_samples.size(), dtype=latent_samples.dtype,
                layout=latent_samples.layout, device="cpu",
            )

        noise_mask = latent.get("noise_mask", None)
        callback = latent_preview.prepare_callback(m, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = comfy.sample.sample(
            m, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_samples,
            denoise=denoise,
            disable_noise=(not add_noise),
            start_step=start_at_step,
            last_step=end_at_step,
            force_full_denoise=(not return_with_leftover_noise),
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise_seed,
        )

        out = latent.copy()
        out["samples"] = samples
        return (out,)


class VaceScheduleModelPatch:
    """Patch a MODEL so per-step VACE strength schedules on conditioning take effect.

    Intended for use with SamplerCustomAdvanced + a custom guider (e.g. KJNodes
    `Guider_ScheduledCFG` for per-step CFG). Equivalent to the VACE-strength portion
    of VaceScheduleKSampler, extracted so it can plug into the modular sampler graph.

    Applies two patches to a cloned model:
    1. `add_object_patch("extra_conds", ...)` — forwards `vace_strength_schedule` (stamped
       by WanVaceToVideoScheduled) through to the forward pass's `c` dict as a CONDConstant.
    2. `set_model_unet_function_wrapper(...)` — before each unet call, bisects the current
       timestep against the given sigma array to find step index, picks `schedule[step_idx]`
       per encoder, rewrites `c["vace_strength"]`, drops `vace_strength_schedule` from `c`.

    Pass the SAME `sigmas` tensor that you pass to SamplerCustomAdvanced. In a split-sampler
    flow, feed the FULL schedule (from BasicScheduler, before SplitSigmas) — that way step
    indices stay consistent across both stages and one N-value VACE schedule can span both.

    This node does NOT handle CFG scheduling (use KJNodes' ScheduledCFGGuidance for that)
    or LoRA scheduling (use LoraStepScheduleLoader — attaches hooks on conditioning,
    sampler-agnostic).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "sigmas": ("SIGMAS",),
            },
            "optional": {
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Log the sigma array and per-step vace_strength resolution to the console.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model"

    def patch(self, model, sigmas, verbose=False):
        m = model.clone()

        sigmas_cpu = sigmas.detach().cpu().clone()
        total_steps = max(1, sigmas_cpu.numel() - 1)
        if verbose:
            log.info(f"[VaceScheduleModelPatch] steps={total_steps} "
                     f"sigmas={[f'{float(s):.4f}' for s in sigmas_cpu.tolist()]}")

        original_extra_conds = m.model.extra_conds
        m.add_object_patch("extra_conds", _make_patched_extra_conds(original_extra_conds))
        m.set_model_unet_function_wrapper(_make_unet_wrapper(sigmas_cpu, total_steps, verbose=verbose))

        return (m,)


class CfgScheduleModelPatch:
    """Patch a MODEL so a per-step CFG schedule takes effect with any sampler.

    Works with stock KSampler, KSamplerAdvanced, SamplerCustomAdvanced, etc. Registers
    a `sampler_cfg_function` that bisects each step's current sigma against the sigma
    array to find the step index, then applies `schedule[step_idx]` as the effective CFG.

    `disable_cfg1_optimization` is enabled automatically when any scheduled value > 1.0
    (so uncond gets computed even when the sampler's scalar cfg is 1.0). When every
    scheduled value is <= 1.0, the optimization is left on and uncond is skipped — saves
    a full forward pass per step.

    IMPORTANT: the SIGMAS you pass in must be the same schedule the sampler runs.
    - For SamplerCustomAdvanced: use the sigmas from BasicScheduler (or SplitSigmas
      for split-sampler flows — use the FULL pre-split sigmas so step indices stay
      consistent across stages).
    - For stock KSampler/KSamplerAdvanced: wire a BasicScheduler with the same
      sampler_name / scheduler / steps / denoise values as the KSampler node, and
      feed that BasicScheduler's SIGMAS output into this patch.

    On the KSampler node itself, `cfg` becomes largely a fallback:
    - If every scheduled value is 1.0, set KSampler cfg=1.0 — no uncond needed.
    - Otherwise it doesn't matter (set 1.0 for efficiency; our function overrides it).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "sigmas": ("SIGMAS",),
                "cfg_schedule": ("STRING", {
                    "default": "3.5, 3.0, 1.0, 1.0, 1.0, 1.0",
                    "multiline": False,
                    "tooltip": "Per-step CFG, comma-separated. Length should match steps (len(sigmas)-1). Ignored if cfg_list socket is connected.",
                }),
            },
            "optional": {
                "cfg_list": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Right-click → 'Convert Widget to Input', connect KJNodes' 'String to Float List' (or similar). List overrides cfg_schedule string; scalar widget value is ignored.",
                }),
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Log the schedule, sigmas, and per-step resolved CFG to the console.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model"

    def patch(self, model, sigmas, cfg_schedule, cfg_list=None, verbose=False):
        schedule = _resolve_schedule(cfg_schedule, cfg_list)
        if not schedule:
            if verbose:
                log.info("[CfgScheduleModelPatch] empty schedule -> no-op")
            return (model,)

        m = model.clone()
        sigmas_cpu = sigmas.detach().cpu().clone()
        total_steps = max(1, sigmas_cpu.numel() - 1)

        needs_uncond = any(v > 1.0 for v in schedule)

        if verbose:
            log.info(f"[CfgScheduleModelPatch] schedule={schedule} total_steps={total_steps} "
                     f"disable_cfg1_optimization={needs_uncond}")
            log.info(f"[CfgScheduleModelPatch] sigmas={[f'{float(s):.4f}' for s in sigmas_cpu.tolist()]}")

        m.set_model_sampler_cfg_function(
            _make_cfg_function(schedule, sigmas_cpu, total_steps, verbose=verbose),
            disable_cfg1_optimization=needs_uncond,
        )
        return (m,)


class LoraStepScheduleLoader:
    """Attach a per-step model-side LoRA schedule to conditioning using native hooks.

    Uses comfy.hooks: creates a WeightHook for the LoRA, builds a per-step
    HookKeyframeGroup, and stamps the combined hooks onto positive and negative
    conditioning via set_hooks_for_conditioning. The sampler auto-registers
    these hooks on the model patcher and scales the LoRA delta per step using
    the current keyframe strength (sampler_helpers.py:193).

    CLIP-side LoRA strength is NOT scheduled (CLIP runs once before sampling).
    Wire a normal LoraLoader upstream if you also want CLIP-side LoRA effect.
    """

    def __init__(self):
        self._cached = None  # (path, state_dict)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_schedule": ("STRING", {
                    "default": "1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
                    "multiline": False,
                    "tooltip": "Per-step model-side LoRA strengths, comma-separated. Ignored if strength_list socket is connected.",
                }),
            },
            "optional": {
                "prev_hooks": ("HOOKS",),
                "strength_list": ("FLOAT", {
                    "default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01,
                    "tooltip": "Right-click widget → 'Convert Widget to Input', then connect a FLOAT-list node (KJNodes 'String to Float List', etc.). List overrides strength_schedule; scalar widget value is ignored.",
                }),
                "sigmas": ("SIGMAS", {
                    "tooltip": "OPTIONAL but strongly recommended for split samplers or non-linear schedulers. Connect the same SIGMAS you feed to SamplerCustomAdvanced (from BasicScheduler). Enables step-exact keyframe placement — keyframe[i] fires at actual step i, not at linear sigma-percent i/(n-1). Without this, schedules like [0,0,1,1,0,0] can fire on wrong steps when sigmas are non-uniform (e.g. beta scheduler).",
                }),
                "model": ("MODEL", {
                    "tooltip": "OPTIONAL. When both sigmas and model are connected, uses model.model_sampling.shift for exact sigma→percent inversion. Without model, assumes shift=1 (typical for WAN flow) and uses the linear formula (1 - sigma/sigma_max). For WAN 2.1/2.2 default configs either path yields the same result; the model input just adds precision for shifted flows.",
                }),
                "sparse_keyframes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Skip emitting a keyframe when the strength matches the previous one. Reduces per-step weight re-patching on schedules with repeated values, e.g. [0, 0, 1, 1, 0, 0] -> 3 keyframes instead of 6. Behaviourally identical.",
                }),
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Log the parsed schedule and keyframe layout (strength, start_percent, target sigma) to the ComfyUI console.",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "HOOKS")
    RETURN_NAMES = ("positive", "negative", "hooks")
    FUNCTION = "apply"
    CATEGORY = "advanced/hooks/scheduling"

    def _load_lora(self, lora_name):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if self._cached is not None and self._cached[0] == lora_path:
            return self._cached[1]
        sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        self._cached = (lora_path, sd)
        return sd

    @staticmethod
    def _sigmas_to_percents(sigmas, model=None):
        """Compute sigma-percent for each step based on the actual sigma array.

        Returns one percent per step (so len-1 values for a sigmas tensor of length N+1).

        Hook keyframes fire when sigma crosses `model_sampling.percent_to_sigma(start_percent)`.
        For WAN flow matching (shift=1): percent_to_sigma(p) = 1 - p, so percent = 1 - sigma.
        For shift != 1: analytical inverse of time_snr_shift.

        Without this, keyframes at linear i/(n-1) percents can land on wrong steps when the
        sampler's sigmas are non-uniform (e.g. `beta` scheduler on a 6-step split).
        """
        sigmas_cpu = sigmas.detach().cpu().flatten()
        n_steps = max(1, sigmas_cpu.numel() - 1)
        shift = 1.0
        if model is not None:
            try:
                shift = float(getattr(model.get_model_object("model_sampling"), "shift", 1.0))
            except Exception:
                shift = 1.0
        percents = []
        for i in range(n_steps):
            s = float(sigmas_cpu[i])
            if s >= 1.0:
                p = 0.0
            elif s <= 0.0:
                p = 1.0
            elif shift == 1.0:
                p = 1.0 - s
            else:
                # Inverse of time_snr_shift: t = s / (shift - s*(shift-1))
                t = s / max(1e-8, shift - s * (shift - 1.0))
                p = 1.0 - t
            percents.append(min(1.0, max(0.0, p)))
        return percents

    def apply(self, positive, negative, lora_name, strength_schedule,
              prev_hooks=None, strength_list=None, sigmas=None, model=None,
              sparse_keyframes=True, verbose=False):
        schedule = _resolve_schedule(strength_schedule, strength_list)
        if not schedule:
            schedule = [1.0]

        lora_sd = self._load_lora(lora_name)

        my_hooks = comfy.hooks.create_hook_lora(
            lora=lora_sd, strength_model=1.0, strength_clip=0.0
        )

        kf_group = comfy.hooks.HookKeyframeGroup()
        n = len(schedule)

        step_percents = None
        if sigmas is not None:
            step_percents = self._sigmas_to_percents(sigmas, model=model)
            if verbose:
                log.info(f"[LoraStepSchedule] lora={lora_name} schedule={schedule} sparse={sparse_keyframes} "
                         f"step-exact (sigmas provided, {len(step_percents)} percents)")
        elif verbose:
            log.info(f"[LoraStepSchedule] lora={lora_name} schedule={schedule} sparse={sparse_keyframes} "
                     f"linear-percent fallback (no sigmas input)")

        def pct_for(i):
            if step_percents is not None:
                idx = min(i, len(step_percents) - 1)
                return step_percents[idx]
            return float(i) / max(1, n - 1) if n > 1 else 0.0

        prev_strength = None
        emitted = 0
        skipped = 0
        for i, s in enumerate(schedule):
            pct = pct_for(i)
            s_val = float(s)
            if sparse_keyframes and prev_strength is not None and s_val == prev_strength:
                skipped += 1
                if verbose:
                    log.info(f"[LoraStepSchedule]   keyframe[{i}] strength={s_val:.3f} start_percent={pct:.3f} (skipped, same as prev)")
                continue
            kf_group.add(comfy.hooks.HookKeyframe(
                strength=s_val, start_percent=pct, guarantee_steps=1,
            ))
            prev_strength = s_val
            emitted += 1
            if verbose:
                log.info(f"[LoraStepSchedule]   keyframe[{i}] strength={s_val:.3f} start_percent={pct:.3f} guarantee_steps=1")

        if verbose:
            log.info(f"[LoraStepSchedule] emitted {emitted} keyframes (skipped {skipped} redundant)")

        my_hooks = my_hooks.clone()
        my_hooks.set_keyframes_on_hooks(kf_group)

        if prev_hooks is not None:
            combined = prev_hooks.clone_and_combine(my_hooks)
        else:
            combined = my_hooks

        positive_out = comfy.hooks.set_hooks_for_conditioning(positive, combined)
        negative_out = comfy.hooks.set_hooks_for_conditioning(negative, combined)

        return (positive_out, negative_out, combined)


class VaceAutoTrimLatent:
    """Trim the reference-image frames off a sampled latent, reading the count from conditioning.

    `WanVaceToVideoScheduled` stamps `vace_trim_latent` (per-encoder int) onto positive and
    negative conditioning via append=True. This node walks those stamps, takes the max across
    all encoders on a cond entry, and slices that many frames off the front of the latent's
    temporal dim. Place this between your last sampler and the VAE decode.

    No extra wiring required beyond `latent` + `conditioning`. Ignores ConditioningSetTimestepRange
    windowing — trim is based on how the ref frames were prepended at encode time, which is
    independent of sigma windows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "conditioning": ("CONDITIONING",),
            },
            "optional": {
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Log the detected trim count and resulting latent T to the console.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "trim_applied")
    FUNCTION = "trim"
    CATEGORY = "latent/video"

    @staticmethod
    def _max_trim_from_cond(cond_list):
        max_trim = 0
        if not cond_list:
            return max_trim
        for entry in cond_list:
            if isinstance(entry, (list, tuple)) and len(entry) > 1 and isinstance(entry[1], dict):
                for t in entry[1].get("vace_trim_latent", []):
                    try:
                        t = int(t)
                    except (TypeError, ValueError):
                        continue
                    if t > max_trim:
                        max_trim = t
        return max_trim

    def trim(self, latent, conditioning, verbose=False):
        trim = self._max_trim_from_cond(conditioning)
        samples = latent["samples"]
        if trim <= 0 or trim >= samples.shape[2]:
            if verbose:
                log.info(f"[VaceAutoTrim] trim={trim} latent_T={samples.shape[2]} -> passthrough")
            out = latent.copy()
            out["samples"] = samples
            return (out, 0)
        trimmed = samples[:, :, trim:, :, :].contiguous()
        if verbose:
            log.info(f"[VaceAutoTrim] trim={trim} latent_T {samples.shape[2]} -> {trimmed.shape[2]}")
        out = latent.copy()
        out["samples"] = trimmed
        if "noise_mask" in out and out["noise_mask"] is not None:
            nm = out["noise_mask"]
            if nm.ndim >= 3 and nm.shape[-3] == samples.shape[2]:
                out["noise_mask"] = nm[..., trim:, :, :].contiguous()
        return (out, trim)


class MaskPrependRefFrame:
    """Prepend zero pixel frames to a mask to match a VACE +ref-slot latent.

    When `LatentPrependRefFrame` adds `trim_latent` ref slots to the front of the
    source latent, the noise mask also needs leading zero frames so that ComfyUI's
    trilinear `reshape_mask` (used by `Set Latent Noise Mask` and `LatentNoiseMaskRebase`)
    aligns the ref-slot position to mask=0 (preserve) rather than bleeding the user's
    frame-0 mask values into the ref slot.

    Pad amount is `trim_latent * temporal_ratio` zero frames at the front.
    For WAN 2.x VAE the temporal ratio is 4 (default).

    Example: 201-frame mouth mask + trim_latent=1 → 205-frame padded mask.
    Trilinear resize to 52 latent frames puts zeros at latent frame 0 (ref slot)
    and the user's mask values at latent frames 1..51.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "trim_latent": ("INT", {"default": 1, "min": 0, "max": 16}),
            },
            "optional": {
                "temporal_ratio": ("INT", {
                    "default": 4, "min": 1, "max": 16,
                    "tooltip": "Pixel frames per latent frame for the model's VAE. WAN 2.x = 4."
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "pad"
    CATEGORY = "latent/video"

    def pad(self, mask, trim_latent, temporal_ratio=4):
        if trim_latent <= 0:
            return (mask,)

        m = mask
        if m.ndim == 2:
            m = m.unsqueeze(0)
        if m.ndim != 3:
            raise ValueError(
                f"[MaskPrependRefFrame] expected 2D [H,W] or 3D [F,H,W] mask, got {tuple(m.shape)}"
            )

        pad_frames = trim_latent * temporal_ratio
        F_in, H, W = m.shape
        zero_pad = torch.zeros((pad_frames, H, W), device=m.device, dtype=m.dtype)
        new_mask = torch.cat([zero_pad, m], dim=0)
        return (new_mask,)


class LatentPrependRefFrame:
    """Prepend reference-image slots to a source latent so it matches a VACE-encoded latent's shape.

    `WanVaceToVideoScheduled` returns a `trim_latent` int (number of ref frames it prepended
    at the front of its sampling latent). When you VAE-encode a source video separately to
    use as a noise-mask rebase anchor, the encoded source is too short by `trim_latent` frames
    to match the VACE latent's shape — which `LatentNoiseMaskRebase` requires for elementwise
    blending.

    This node prepends the missing slots. Two modes:

      - `zeros`: fill the prepended slots with zeros. Matches what the VACE encoder writes
        at the ref position in its own sampling latent (the ref image rides on conditioning,
        not on the sampling latent's content).

      - `encoded_ref`: fill with frames from a separately-VAE-encoded reference image.
        Slightly nicer for the model since the prepended slot has coherent content, but
        for full-denoise rebase flows the difference is cosmetic — the slot gets fully
        denoised regardless and is trimmed by `VaceAutoTrimLatent` before decode.

    Wiring:

        Source Video -> VAEEncode -> source_latent (51F)
        WanVaceToVideoScheduled -> ..., trim_latent (=1 if ref present)
        [optional] Ref Image -> VAEEncode -> ref_latent (1F)

        LatentPrependRefFrame(source_latent, trim_latent, ref_latent?) -> padded_source (52F)

        LatentNoiseMaskRebase(prev_latent, padded_source, mask) -> rebased
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_latent": ("LATENT",),
                "trim_latent": ("INT", {"default": 1, "min": 0, "max": 16}),
            },
            "optional": {
                "pad_mode": (["zeros", "encoded_ref"], {"default": "zeros"}),
                "ref_latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "pad"
    CATEGORY = "latent/video"

    def pad(self, source_latent, trim_latent, pad_mode="zeros", ref_latent=None):
        s = source_latent["samples"]

        if s.dim() != 5:
            raise ValueError(
                f"[LatentPrependRefFrame] expected 5D latent [B,C,F,H,W], got {tuple(s.shape)}"
            )

        if trim_latent <= 0:
            return (source_latent,)

        B, C, F_src, H, W = s.shape

        if pad_mode == "encoded_ref":
            if ref_latent is None:
                log.warning(
                    "[LatentPrependRefFrame] pad_mode=encoded_ref but no ref_latent connected; "
                    "falling back to zeros."
                )
                ref_pad = torch.zeros((B, C, trim_latent, H, W), device=s.device, dtype=s.dtype)
            else:
                r = ref_latent["samples"]
                if r.dim() != 5:
                    raise ValueError(
                        f"[LatentPrependRefFrame] ref_latent must be 5D, got {tuple(r.shape)}"
                    )
                if r.shape[1] != C or r.shape[3] != H or r.shape[4] != W:
                    raise ValueError(
                        f"[LatentPrependRefFrame] ref_latent C/H/W mismatch: "
                        f"source has C={C} H={H} W={W}; "
                        f"ref has C={r.shape[1]} H={r.shape[3]} W={r.shape[4]}"
                    )
                if r.shape[2] < trim_latent:
                    raise ValueError(
                        f"[LatentPrependRefFrame] ref_latent has {r.shape[2]} frames, "
                        f"need at least trim_latent={trim_latent}"
                    )
                ref_pad = r[:, :, :trim_latent].to(device=s.device, dtype=s.dtype)
                if ref_pad.shape[0] != B:
                    if ref_pad.shape[0] == 1:
                        ref_pad = ref_pad.expand(B, -1, -1, -1, -1).contiguous()
                    else:
                        raise ValueError(
                            f"[LatentPrependRefFrame] ref_latent batch={ref_pad.shape[0]} "
                            f"can't broadcast to source batch={B}"
                        )
        else:
            ref_pad = torch.zeros((B, C, trim_latent, H, W), device=s.device, dtype=s.dtype)

        new_samples = torch.cat([ref_pad, s], dim=2)

        out = source_latent.copy()
        out["samples"] = new_samples
        return (out,)


class LatentNoiseMaskInspect:
    """Diagnostic passthrough: log a latent's shape and noise_mask state.

    Drop between samplers when debugging chained `Set Latent Noise Mask` flows.
    Logs samples shape/dtype/device, plus whether `noise_mask` survived the prior
    sampler and what shape it's in. Output latent is identical to input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "label": ("STRING", {"default": "latent", "multiline": False}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "inspect"
    CATEGORY = "latent/video"

    def inspect(self, latent, label):
        s = latent.get("samples")
        nm = latent.get("noise_mask")
        if s is None:
            log.info(f"[LatentInspect:{label}] samples=None")
        else:
            log.info(
                f"[LatentInspect:{label}] samples shape={tuple(s.shape)} "
                f"dtype={s.dtype} device={s.device}"
            )
        if nm is None:
            log.info(f"[LatentInspect:{label}] noise_mask=None (the chain has dropped it)")
        else:
            try:
                stats = (
                    f"min={nm.min().item():.3f} max={nm.max().item():.3f} "
                    f"mean={nm.float().mean().item():.3f}"
                )
            except Exception:
                stats = "(stats unavailable)"
            log.info(
                f"[LatentInspect:{label}] noise_mask shape={tuple(nm.shape)} "
                f"dtype={nm.dtype} device={nm.device} {stats}"
            )
        return (latent,)


class LatentNoiseMaskRebase:
    """Re-anchor a noise-masked latent chain to the original source between samplers.

    What this does (latent content):
      - Masked region (mask=1, "regenerate"): keep the previous sampler's in-flight latent.
      - Unmasked region (mask=0, "preserve"): replace with source content, optionally
        pre-noised in CONST inverse-scaled form so the next sampler's `noise_scaling`
        produces a correctly-distributed `x at sigma`.

    Why pre-noising matters (CONST / flow-matching, e.g. WAN):

        The previous sampler's output gets `inverse_noise_scaling` applied at exit:
            output = (sigma*noise + (1-sigma)*x_0) / (1-sigma) = sigma/(1-sigma)*noise + x_0

        The next sampler runs `noise_scaling(sigma, noise_obj, latent_image)` to rebuild
        x at sigma:
            x_initial = sigma * noise_obj + (1-sigma) * latent_image

        With DisableNoise (noise_obj = 0), x_initial = (1-sigma) * latent_image.
        For the masked region, latent_image is the prev sampler's inverse-scaled output,
        and the math collapses to the correct CONST form: sigma*noise_prev + (1-sigma)*x_0.

        For the unmasked region, if we put plain source there, we get (1-sigma)*source —
        no noise component. Out of distribution at sigma > 0, which corrupts the model's
        attention and produces brown/grey blobs in the masked region (the model attends
        across all positions; OOD unmasked context distorts masked denoising).

        Fix: put `source + sigma/(1-sigma)*noise` in unmasked. After noise_scaling at
        the next sampler:
            x_initial[unmasked] = (1-sigma) * (source + sigma/(1-sigma)*noise)
                                = (1-sigma)*source + sigma*noise            ✓ correct CONST

    Inputs:
      - `next_sampler_sigmas` (SIGMAS, optional): when connected, pre-noise the source
        using sigmas[0] of the next sampler. Leave disconnected to skip pre-noising
        (use clean source — only correct if the next sampler is sigma=0 i.e. final).
      - `noise_seed`: seed for the noise added to source. Use the same seed across all
        rebase calls in the chain so the unmasked region has a consistent noise pattern
        across samplers.
      - `reattach_noise_mask` (default False): see option A vs B discussion. Leave off
        for DisableNoise chains.

    Polarity: matches `Set Latent Noise Mask` — mask=1 = regenerate, mask=0 = preserve.
    Source latent shape must match prev_latent shape exactly (use `LatentPrependRefFrame`
    if VACE added a ref slot).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prev_latent": ("LATENT",),
                "source_latent": ("LATENT",),
                "mask": ("MASK",),
            },
            "optional": {
                "next_sampler_sigmas": ("SIGMAS",),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "reattach_noise_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Re-attach noise_mask for the next sampler. Leave OFF when "
                               "the next sampler uses DisableNoise (CONST-flow correctness).",
                }),
                "verbose": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "rebase"
    CATEGORY = "latent/video"

    def rebase(self, prev_latent, source_latent, mask,
               next_sampler_sigmas=None, noise_seed=0,
               reattach_noise_mask=False, verbose=False):
        prev = prev_latent["samples"]
        src = source_latent["samples"]

        if prev.shape != src.shape:
            raise ValueError(
                f"[LatentNoiseMaskRebase] shape mismatch: prev_latent={tuple(prev.shape)} "
                f"vs source_latent={tuple(src.shape)}. The source must be VAE-encoded at "
                f"the same dims/length as the VACE encoder used (incl. +1 ref frame if any)."
            )

        src_aligned = src.to(device=prev.device, dtype=prev.dtype)

        sigma_used = None
        if next_sampler_sigmas is not None and next_sampler_sigmas.numel() > 0:
            sigma = float(next_sampler_sigmas[0].item())
            eps = 1e-6
            if eps < sigma < 1.0 - eps:
                # CONST flow with per-channel latent normalization (WAN21):
                # sampler's noise object is unit-randn in NORMALIZED space; latent_image
                # gets process_latent_in applied at sampler entry: norm = (raw - mean) / std.
                # We want, after process_latent_in + noise_scaling with DisableNoise:
                #     x_initial[unmasked] = sigma * randn + (1-sigma) * source_normalized
                # Working backwards, the rebased latent (raw) must be:
                #     rebased[unmasked]_raw = source_raw + sigma/(1-sigma) * randn * std
                # Without the * std factor, the noise gets divided by std after process_in,
                # producing under-noised channels (high-std channels carry color → brown bias).
                gen = torch.Generator(device="cpu").manual_seed(int(noise_seed))
                noise_cpu = torch.randn(prev.shape, generator=gen, dtype=torch.float32)
                noise = noise_cpu.to(device=prev.device, dtype=prev.dtype)

                lf = comfy.latent_formats.Wan21()
                std_t = lf.latents_std.to(device=prev.device, dtype=prev.dtype)

                scale = sigma / (1.0 - sigma)
                src_aligned = src_aligned + scale * noise * std_t
                sigma_used = sigma

        m = comfy.utils.reshape_mask(mask, prev.shape).to(device=prev.device, dtype=prev.dtype)
        new_samples = m * prev + (1.0 - m) * src_aligned

        out = prev_latent.copy()
        out["samples"] = new_samples
        if reattach_noise_mask:
            out["noise_mask"] = mask
        else:
            out.pop("noise_mask", None)

        if verbose:
            sigma_str = f"sigma={sigma_used:.4f} (pre-noised)" if sigma_used is not None else "no pre-noise"
            log.info(
                f"[LatentNoiseMaskRebase] rebased shape={tuple(new_samples.shape)} "
                f"mask_reshaped_to={tuple(m.shape)} mask_mean={m.mean().item():.3f} "
                f"{sigma_str} reattach_noise_mask={reattach_noise_mask}"
            )
        return (out,)


NODE_CLASS_MAPPINGS = {
    "WanVaceToVideoScheduled": WanVaceToVideoScheduled,
    "VaceScheduleKSampler": VaceScheduleKSampler,
    "VaceScheduleModelPatch": VaceScheduleModelPatch,
    "CfgScheduleModelPatch": CfgScheduleModelPatch,
    "LoraStepScheduleLoader": LoraStepScheduleLoader,
    "VaceAutoTrimLatent": VaceAutoTrimLatent,
    "LatentPrependRefFrame": LatentPrependRefFrame,
    "MaskPrependRefFrame": MaskPrependRefFrame,
    "LatentNoiseMaskInspect": LatentNoiseMaskInspect,
    "LatentNoiseMaskRebase": LatentNoiseMaskRebase,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVaceToVideoScheduled": "WAN VACE To Video (Step Schedule)",
    "VaceScheduleKSampler": "KSampler (VACE + CFG Step Schedule)",
    "VaceScheduleModelPatch": "VACE Schedule Model Patch (for SamplerCustomAdvanced)",
    "CfgScheduleModelPatch": "CFG Schedule Model Patch (any sampler)",
    "LoraStepScheduleLoader": "LoRA (Step Schedule)",
    "VaceAutoTrimLatent": "VACE Auto-Trim Latent",
    "LatentPrependRefFrame": "Latent Prepend Ref Frame (for rebase)",
    "MaskPrependRefFrame": "Mask Prepend Ref Frame (for rebase)",
    "LatentNoiseMaskInspect": "Latent Noise Mask Inspect (debug)",
    "LatentNoiseMaskRebase": "Latent Noise Mask Rebase (chain anchor)",
}
