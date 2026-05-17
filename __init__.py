from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


# --- Mid-sampling preview: rotate through latent frames each call ---
# ComfyUI's Latent2RGBPreviewer and TAEHVPreviewerImpl both decode frame 0 of the
# video latent. With VACE workflows that prepend a reference image as the first
# latent slot, frame 0 is the ref and a single fixed mid-frame still looks static
# across sampling steps. We rotate through frames 1..T-1 each preview call so the
# preview shows motion (and skips the ref slot).
def _install_rotating_frame_preview():
    import logging
    log = logging.getLogger(__name__)
    try:
        import latent_preview as lp

        state = {"cursor": 1, "shape_logged": None}

        def _pick_frame(x0, src):
            shape_t = tuple(x0.shape)
            if state["shape_logged"] != shape_t:
                state["shape_logged"] = shape_t
                log.info(f"[vace_step_schedule] {src} preview x0.shape={shape_t}")
            if x0.ndim == 5 and x0.shape[2] > 1:
                T = x0.shape[2]
                idx = state["cursor"]
                if idx < 1 or idx >= T:
                    idx = 1
                state["cursor"] = (idx + 1) if (idx + 1) < T else 1
                x0 = x0[:, :, idx:idx + 1]
            return x0

        orig_l2rgb = lp.Latent2RGBPreviewer.decode_latent_to_preview
        def patched_l2rgb(self, x0):
            return orig_l2rgb(self, _pick_frame(x0, "L2RGB"))
        lp.Latent2RGBPreviewer.decode_latent_to_preview = patched_l2rgb

        orig_taehv = lp.TAEHVPreviewerImpl.decode_latent_to_preview
        def patched_taehv(self, x0):
            return orig_taehv(self, _pick_frame(x0, "TAEHV"))
        lp.TAEHVPreviewerImpl.decode_latent_to_preview = patched_taehv

        log.info("[vace_step_schedule] preview patch: rotating through latent frames 1..T-1 (skip ref at 0)")
    except Exception as e:
        log.warning(f"[vace_step_schedule] preview patch failed: {e}")

_install_rotating_frame_preview()
