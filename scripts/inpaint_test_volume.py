import numpy as np
import torch

from utils.logging import log


def inpaint_volume_with_model(model, gt_volume, corrupted_volume, device="cuda", stack_size=9, debug=False, args=None):
    """
    Run single-pass slice-wise inpainting over empty test slices.

    Args:
        model: Trained inpainting model.
        gt_volume: Retained for compatibility with the current pipeline; unused here.
        corrupted_volume: (D, H, W) test volume loaded from disk.
        stack_size: Number of slices in the input stack.
        args: Retained for compatibility with the current pipeline; unused here.

    Returns:
        np.ndarray: Inpainted volume with the same shape as the input volume.
    """
    model.eval()
    pad = stack_size // 2

    log("[DEBUG] Only inpainting empty slices...")
    inpaint_mask = (corrupted_volume.mean(axis=(1, 2)) == 0).astype(np.uint8)

    if debug:
        log(f"[DEBUG] Inpainting slices detected: {np.where(inpaint_mask == 1)[0].tolist()}")

    corrupted_volume = corrupted_volume.copy()
    intensity_scale = float(corrupted_volume.max()) + 1e-5
    corrupted_volume[inpaint_mask == 1] = 0

    padded = np.pad(corrupted_volume, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    inpainted = corrupted_volume.copy()

    with torch.no_grad():
        for idx in np.where(inpaint_mask == 1)[0]:
            stack = padded[idx: idx + stack_size]

            stack_tensor = torch.from_numpy(stack).unsqueeze(0).float().to(device) / intensity_scale

            _valid_mask = (stack_tensor.squeeze(0).sum(dim=(1, 2)) > 1e-3).float()
            _valid_mask = _valid_mask.unsqueeze(0).to(device)

            output = model(stack_tensor)

            pred = output.squeeze().cpu().numpy() * intensity_scale

            inpainted[idx] = np.clip(pred, 0, corrupted_volume.max()).astype(np.uint16)

    return inpainted
