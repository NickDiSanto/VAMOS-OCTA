import torch
import numpy as np
from tqdm import tqdm

from utils.logging import log


def evaluate_model_on_test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (X, y, valid_mask) in enumerate(tqdm(dataloader, desc="Testing")):
            X, y, valid_mask = X.to(device), y.to(device), valid_mask.to(device)

            X = X.contiguous().float()
            y = y.contiguous().float()

            output = model(X)

            if torch.isnan(output).any() or torch.isinf(output).any():
                log(f"[ERROR] Model output contains NaNs or Infs at batch {batch_idx}")

            loss, _ = criterion(output, y)

            running_loss += loss.item() * X.size(0)

    return running_loss / len(dataloader.dataset)


def detect_inpainting_slices(volume, threshold=0.25, margin=0.0):
    """
    Detect corrupted slices in a volume based on brightness deviation from median.
    
    Args:
        volume (np.ndarray): (D, H, W) uint16 volume.
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
    Apply 2.5D model to slices automatically detected as corrupted.
    Slices deviating from the median brightness are selected and zeroed out before being passed to the model.
    
    Args:
        model: trained UNet2p5D model
        corrupted_volume: (D, H, W) array (uint16)
        stack_size: number of slices in input stack

    Returns:
        np.ndarray: inpainted volume (D, H, W), dtype uint16
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

    return inpainted.astype(np.uint16)
