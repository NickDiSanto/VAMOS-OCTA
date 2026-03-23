import torch
import numpy as np

from utils.logging import log


def inpaint_volume_with_model(model, gt_volume, corrupted_volume, device='cuda', stack_size=9, debug=False, args=None):
    """
    Apply model to corrupted slices, which are zeroed out before being passed to the model.

    Args:
        model: trained model
        corrupted_volume: (D, H, W) array (float32)
        stack_size: number of slices in input stack

    Returns:
        np.ndarray: inpainted volume (D, H, W), dtype float32
    """
    model.eval()
    pad = stack_size // 2

    log("[DEBUG] Only inpainting empty slices...")
    mask = (corrupted_volume.mean(axis=(1, 2)) == 0).astype(np.uint8)

    if debug:
        log(f"[DEBUG] Inpainting slices detected: {np.where(mask == 1)[0].tolist()}")

    # Zero out slices flagged for inpainting
    corrupted_volume = corrupted_volume.copy()
    intensity_scale = float(corrupted_volume.max()) + 1e-5
    corrupted_volume[mask == 1] = 0

    padded = np.pad(corrupted_volume, ((pad, pad), (0, 0), (0, 0)), mode='edge')
    inpainted = corrupted_volume.copy()

    with torch.no_grad():
        for idx in np.where(mask == 1)[0]:
            stack = padded[idx: idx + stack_size]  # (stack_size, H, W)

            stack_tensor = torch.from_numpy(stack).unsqueeze(0).float().to(device) / intensity_scale

            valid_mask = (stack_tensor.squeeze(0).sum(dim=(1, 2)) > 1e-3).float()  # shape: (stack_size,)
            valid_mask = valid_mask.unsqueeze(0).to(device)  # shape: (1, stack_size)

            output = model(stack_tensor)

            pred = output.squeeze().cpu().numpy() * intensity_scale

            inpainted[idx] = np.clip(pred, 0, corrupted_volume.max()).astype(np.uint16)

    return inpainted
