import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim

from evaluation.metrics_utils import compute_local_ncc
from utils.logging import log


def visualize_ncc_slice_stacked(gt_volume, pred_volume, slice_idx, window_sizes=[7, 11, 17]):
    """Visualize GT, prediction, and NCC heatmaps for one slice."""
    gt = gt_volume[slice_idx].astype(np.float32)
    pred = pred_volume[slice_idx].astype(np.float32)

    num_windows = len(window_sizes)
    nrows = 2 + num_windows
    fig, axes = plt.subplots(nrows, 1, figsize=(5, 2.5 * nrows))
    plt.subplots_adjust(hspace=0.2, top=0.95, bottom=0.05)

    ax_gt = axes[0]
    ax_gt.imshow(gt, cmap="gray")
    ax_gt.set_title("Ground-Truth", fontsize=10, pad=5)
    ax_gt.axis("off")

    ax_pred = axes[1]
    ax_pred.imshow(pred, cmap="gray")
    ax_pred.set_title("Predicted", fontsize=10, pad=5)
    ax_pred.axis("off")

    for i, window_size in enumerate(window_sizes):
        ax_ncc = axes[2 + i]
        try:
            ncc_map = np.clip(compute_local_ncc(gt, pred, window_size=window_size), -1, 1)
        except Exception as e:
            log(f"NCC computation for win_size={window_size} failed: {e}")
            ncc_map = np.zeros_like(gt)

        im = ax_ncc.imshow(ncc_map, cmap="bwr", vmin=-1, vmax=1)
        ax_ncc.set_title(f"NCC Map (win={window_size})", fontsize=10, pad=5)

        cbar = fig.colorbar(im, ax=ax_ncc, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        ax_ncc.axis("off")

    plt.show()


def visualize_ssim_slice_stacked(gt_volume, pred_volume, slice_idx, window_sizes=[7, 11, 17]):
    """Visualize GT, prediction, and SSIM heatmaps for one slice."""
    gt = gt_volume[slice_idx].astype(np.float32)
    pred = pred_volume[slice_idx].astype(np.float32)

    num_windows = len(window_sizes)
    nrows = 2 + num_windows
    fig, axes = plt.subplots(nrows, 1, figsize=(5, 2.5 * nrows))
    plt.subplots_adjust(hspace=0.2, top=0.95, bottom=0.05)

    ax_gt = axes[0]
    ax_gt.imshow(gt, cmap="gray")
    ax_gt.set_title("Ground-Truth", fontsize=10, pad=5)
    ax_gt.axis("off")

    ax_pred = axes[1]
    ax_pred.imshow(pred, cmap="gray")
    ax_pred.set_title("Predicted", fontsize=10, pad=5)
    ax_pred.axis("off")

    for i, window_size in enumerate(window_sizes):
        ax_ssim = axes[2 + i]
        try:
            ssim_map = skimage_ssim(gt, pred, data_range=1.0, win_size=window_size, full=True)[1]
        except Exception as e:
            log(f"SSIM computation for win_size={window_size} failed: {e}")
            ssim_map = np.zeros_like(gt)

        im = ax_ssim.imshow(ssim_map, cmap="hot", vmin=0, vmax=1)
        ax_ssim.set_title(f"SSIM Map (win={window_size})", fontsize=10, pad=5)

        cbar = fig.colorbar(im, ax=ax_ssim, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        ax_ssim.axis("off")

    plt.show()


def visualize_slice_panel(gt_volume, pred_volume, mask, slice_indices=None, ncols=3):
    """Visualize GT, prediction, and error maps for selected masked slices."""
    nrows = 5

    if slice_indices is None:
        slice_indices = np.where(mask)[0][:ncols]

    fig, axes = plt.subplots(nrows, len(slice_indices), figsize=(4 * len(slice_indices), 3.8 * nrows))
    plt.subplots_adjust(hspace=0.6, wspace=0.1)

    row_labels = ["GT", "Prediction", "NCC17", "SSIM11", "Abs Error"]

    for col, idx in enumerate(slice_indices):
        gt = gt_volume[idx].astype(np.float32)
        pred = pred_volume[idx].astype(np.float32)
        abs_error = np.abs(gt - pred)

        ncc_map = np.clip(compute_local_ncc(gt, pred, window_size=17), -1, 1)
        try:
            ssim_map = skimage_ssim(gt, pred, data_range=1.0, win_size=11, full=True)[1]
        except ValueError:
            ssim_map = np.zeros_like(gt)

        slices = [
            (gt, "gray", None),
            (pred, "gray", None),
            (ncc_map, "bwr", (-1, 1)),
            (ssim_map, "hot", (0, 1)),
            (abs_error, "hot", (np.percentile(abs_error, 1), np.percentile(abs_error, 99))),
        ]

        axes[0, col].set_title(f"Slice {idx}", fontsize=12, pad=20)

        for row, (image, cmap, limits) in enumerate(slices):
            ax = axes[row, col]
            vmin, vmax = limits if limits else (None, None)
            im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)

            ax.set_title(row_labels[row], fontsize=10, loc="center", pad=5)

            if row in [2, 3, 4]:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.01)
                cbar.ax.tick_params(labelsize=8)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)

    plt.tight_layout()
    plt.show()
