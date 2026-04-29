from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


# --- Mid-sampling preview: show a middle frame instead of frame 0 ---
# ComfyUI's Latent2RGBPreviewer and TAEHVPreviewerImpl both decode frame 0 of the
# video latent. With VACE workflows that prepend a reference image as the first
# latent slot, frame 0 is always the ref — so the preview never shows real sampled
# content. We monkey-patch both previewers to slice the middle frame instead.
def _install_middle_frame_preview():
    import logging
    log = logging.getLogger(__name__)
    try:
        import latent_preview as lp

        orig_l2rgb = lp.Latent2RGBPreviewer.decode_latent_to_preview
        def patched_l2rgb(self, x0):
            if x0.ndim == 5 and x0.shape[2] > 1:
                mid = x0.shape[2] // 2
                x0 = x0[:, :, mid:mid + 1]
            return orig_l2rgb(self, x0)
        lp.Latent2RGBPreviewer.decode_latent_to_preview = patched_l2rgb

        orig_taehv = lp.TAEHVPreviewerImpl.decode_latent_to_preview
        def patched_taehv(self, x0):
            if x0.ndim >= 4 and x0.shape[2] > 1:
                mid = x0.shape[2] // 2
                x0 = x0[:, :, mid:mid + 1]
            return orig_taehv(self, x0)
        lp.TAEHVPreviewerImpl.decode_latent_to_preview = patched_taehv

        log.info("[vace_step_schedule] preview patch: video previews will show middle frame (skip ref slot)")
    except Exception as e:
        log.warning(f"[vace_step_schedule] preview patch failed: {e}")

_install_middle_frame_preview()
