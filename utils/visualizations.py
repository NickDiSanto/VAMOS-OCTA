import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as skimage_ssim

from evaluation.metrics_utils import compute_local_ncc
from utils.logging import log


def visualize_ncc_slice_stacked(gt_volume, pred_volume, slice_idx, window_sizes=[7, 11, 17]):
    """
    Visualize GT, prediction, and NCC heatmaps for one slice
    with multiple window_sizes stacked vertically.
    """
    gt = (gt_volume[slice_idx].astype(np.float32)) / (gt.max() + 1e-5)
    pred = (pred_volume[slice_idx].astype(np.float32)) / (gt.max() + 1e-5)

    num_windows = len(window_sizes)
    nrows = 2 + num_windows # GT, Pred, then num_windows for NCC maps
    
    # Use the same figure size as your final SSIM function
    fig, axes = plt.subplots(nrows, 1, figsize=(5, 2.5 * nrows))
    
    # Use the same hspace, top, and bottom as your final SSIM function
    plt.subplots_adjust(hspace=0.2, top=0.95, bottom=0.05)

    # Row 0: Ground-Truth Slice
    ax_gt = axes[0]
    ax_gt.imshow(gt, cmap='gray')
    ax_gt.set_title(f"Ground-Truth", fontsize=10, pad=5) # Match SSIM title format
    ax_gt.axis('off')

    # Row 1: Predicted Slice
    ax_pred = axes[1]
    ax_pred.imshow(pred, cmap='gray')
    ax_pred.set_title(f"Predicted", fontsize=10, pad=5) # Match SSIM title format
    ax_pred.axis('off')

    # Subsequent Rows: NCC Maps for different window sizes
    for i, window_size in enumerate(window_sizes):
        ax_ncc = axes[2 + i]
        try:
            # Use your provided compute_local_ncc function
            ncc_map = np.clip(compute_local_ncc(gt, pred, window_size=window_size), -1, 1)
        except Exception as e:
            log(f"NCC computation for win_size={window_size} failed: {e}")
            ncc_map = np.zeros_like(gt) # Fallback if NCC fails

        # NCC values range from -1 to 1, so use 'bwr' cmap and set vmin/vmax accordingly
        im = ax_ncc.imshow(ncc_map, cmap='bwr', vmin=-1, vmax=1)
        # Match SSIM title format: "NCC Map (win=X)"
        ax_ncc.set_title(f"NCC Map (win={window_size})", fontsize=10, pad=5)
        
        # Match colorbar styling
        cbar = fig.colorbar(im, ax=ax_ncc, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        ax_ncc.axis('off')

    plt.show()


def visualize_ssim_slice_stacked(gt_volume, pred_volume, slice_idx, window_sizes=[7, 11, 17]):
    """
    Visualize GT, prediction, and SSIM heatmaps for one slice
    with multiple window_sizes stacked vertically.
    """
    gt = (gt_volume[slice_idx].astype(np.float32)) / (gt.max() + 1e-5)
    pred = (pred_volume[slice_idx].astype(np.float32)) / (gt.max() + 1e-5)

    num_windows = len(window_sizes)
    nrows = 2 + num_windows # GT, Pred, then num_windows for SSIM maps
    
    # Keep the reduced figure size
    fig, axes = plt.subplots(nrows, 1, figsize=(5, 2.5 * nrows))
    
    # Crucial change: Reduce hspace to bring rows closer
    plt.subplots_adjust(hspace=0.2, top=0.95, bottom=0.05) # Reduced hspace from 0.6 to 0.4

    # Row 0: Ground-Truth Slice
    ax_gt = axes[0]
    ax_gt.imshow(gt, cmap='gray')
    ax_gt.set_title(f"Ground-Truth", fontsize=10, pad=5)
    ax_gt.axis('off')

    # Row 1: Predicted Slice
    ax_pred = axes[1]
    ax_pred.imshow(pred, cmap='gray')
    ax_pred.set_title(f"Predicted", fontsize=10, pad=5)
    ax_pred.axis('off')

    # Subsequent Rows: SSIM Maps for different window sizes
    for i, window_size in enumerate(window_sizes):
        ax_ssim = axes[2 + i]
        try:
            ssim_map = skimage_ssim(gt, pred, data_range=1.0, win_size=window_size, full=True)[1]
        except Exception as e:
            log(f"SSIM computation for win_size={window_size} failed: {e}")
            ssim_map = np.zeros_like(gt)

        im = ax_ssim.imshow(ssim_map, cmap='hot', vmin=0, vmax=1)
        ax_ssim.set_title(f"SSIM Map (win={window_size})", fontsize=10, pad=5)
        
        cbar = fig.colorbar(im, ax=ax_ssim, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        ax_ssim.axis('off')

    plt.show()


def visualize_slice_panel(gt_volume, pred_volume, mask, slice_indices=None, ncols=3):
    nrows = 5  # GT, Pred, NCC17, SSIM11, AbsError

    if slice_indices is None:
        slice_indices = np.where(mask)[0][:ncols]

    # Adjust figsize for better readability with titles on top
    fig, axes = plt.subplots(nrows, len(slice_indices), figsize=(4 * len(slice_indices), 3.8 * nrows))
    plt.subplots_adjust(hspace=0.6, wspace=0.1) # Increased hspace for titles

    # Define the labels for each row once
    row_labels = ["GT", "Prediction", "NCC17", "SSIM11", "Abs Error"]

    for col, idx in enumerate(slice_indices):
        gt = (gt_volume[idx].astype(np.float32)) / (gt.max() + 1e-5)
        pred = (pred_volume[idx].astype(np.float32)) / (gt.max() + 1e-5)
        abs_error = np.abs(gt - pred)

        ncc_map = np.clip(compute_local_ncc(gt, pred, window_size=17), -1, 1)
        try:
            ssim_map = skimage_ssim(gt, pred, data_range=1.0, win_size=11, full=True)[1]
        except ValueError:
            ssim_map = np.zeros_like(gt)

        slices = [
            (gt, 'gray', None),
            (pred, 'gray', None),
            (ncc_map, 'bwr', (-1, 1)),
            (ssim_map, 'hot', (0, 1)),
            (abs_error, 'hot', (np.percentile(abs_error, 1), np.percentile(abs_error, 99)))
        ]

        # Set the main column title (Slice X) only once per column, on the very top row
        axes[0, col].set_title(f"Slice {idx}", fontsize=12, pad=20) # Increased pad to move it higher

        for row, (image, cmap, limits) in enumerate(slices):
            ax = axes[row, col]
            vmin, vmax = limits if limits else (None, None)
            im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)

            # Set the label for each individual image (GT, Prediction, NCC17, etc.)
            ax.set_title(row_labels[row], fontsize=10, loc='center', pad=5) # Label on top of each image

            # Add colorbar for NCC, SSIM, Error
            if row in [2, 3, 4]:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.01)
                cbar.ax.tick_params(labelsize=8)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.show()
