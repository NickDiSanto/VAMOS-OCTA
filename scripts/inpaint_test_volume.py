import torch
import numpy as np

from utils.logging import log


def detect_inpainting_slices(volume, threshold=0.25, margin=0.0):
    """
    Detect corrupted slices in a volume based on brightness deviation from median.
    
    Args:
        volume (np.ndarray): (D, H, W) float32 volume.
        threshold (float): Relative deviation from median to mark corruption.
        margin (float): Min contrast margin to avoid flat regions.

    Returns:
        np.ndarray: (D,) binary mask where 1 = inpaint this slice.
    """
    slice_brightness = volume.mean(axis=(1, 2))  # shape: (D,)
    median = np.median(slice_brightness)
    deviation = np.abs(slice_brightness - median) / (median + 1e-8)

    mask = deviation > threshold
    mask &= (slice_brightness > median * margin) | (slice_brightness < median * (1 - margin))
    return mask.astype(np.uint8)


def inpaint_volume_with_model(model, gt_volume, corrupted_volume, detect_bright_outliers=False, device='cuda', stack_size=9, debug=False, args=None):
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

    if detect_bright_outliers:
        # Automatically detect bright slices
        log("[DEBUG] Auto-detecting bright outlier slices...")
        mask = detect_inpainting_slices(volume=corrupted_volume, threshold=0.165, margin=0.0)
    else:
        # Only inpaint empty slices
        log("[DEBUG] Only inpainting empty slices...")
        mask = (corrupted_volume.mean(axis=(1, 2)) == 0).astype(np.uint8)

    if debug:
        log(f"[DEBUG] Inpainting slices detected: {np.where(mask == 1)[0].tolist()}")

    # Zero out slices flagged for inpainting
    corrupted_volume = corrupted_volume.copy()
    corrupted_volume[mask == 1] = 0

    padded = np.pad(corrupted_volume, ((pad, pad), (0, 0), (0, 0)), mode='edge')
    inpainted = corrupted_volume.copy()

    with torch.no_grad():
        for idx in np.where(mask == 1)[0]:
            stack = padded[idx: idx + stack_size]  # (stack_size, H, W)

            stack_tensor = torch.from_numpy(stack).unsqueeze(0).float().to(device) / (gt_volume.max() + 1e-5)

            valid_mask = (stack_tensor.squeeze(0).sum(dim=(1, 2)) > 1e-3).float()  # shape: (stack_size,)
            valid_mask = valid_mask.unsqueeze(0).to(device)  # shape: (1, stack_size)

            output = model(stack_tensor)

            pred = output.squeeze().cpu().numpy() * gt_volume.max()

            inpainted[idx] = np.clip(pred, 0, corrupted_volume.max()).astype(np.uint16)

    return inpainted.astype(np.float32)  # Return as float32 for consistency with other processing steps
