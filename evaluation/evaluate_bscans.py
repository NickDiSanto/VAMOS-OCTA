import io
import warnings
from contextlib import redirect_stdout

import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity as skimage_ssim

from evaluation.metrics_utils import (
    compute_local_ncc,
    edge_strength,
    gradient_magnitude,
    laplacian_blur_score,
    to_lpips_tensor,
)
from utils.logging import log


def _build_lpips_model():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The parameter 'pretrained' is deprecated since 0.13.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="You are using `torch.load` with `weights_only=False`.*",
            category=FutureWarning,
        )
        with redirect_stdout(io.StringIO()):
            model = lpips.LPIPS(net="alex")

    return model.cuda() if torch.cuda.is_available() else model


def evaluate_bscans(gt_volume, pred_volume, mask):
    """Evaluate slice-wise reconstruction quality on masked B-scans only."""
    assert gt_volume.shape == pred_volume.shape, "Volume shapes must match"

    num_slices, _, _ = gt_volume.shape
    masked_indices = np.where(mask)[0]
    if len(masked_indices) == 0:
        log("\nB-SCAN METRICS:")
        log(" - No masked slices found; skipping B-scan metric computation.")
        return {
            "L1": float("nan"),
            "MeanIntensityError": float("nan"),
            "PSNR": float("nan"),
            "SSIM11": float("nan"),
            "Global_NCC": float("nan"),
            "Gradient_L1": float("nan"),
            "LPIPS": float("nan"),
            "Edge_Preservation_Ratio": float("nan"),
            "Laplacian_Blur_Score_Diff": float("nan"),
        }

    l1_vals = []
    psnr_vals = []
    mie_vals = []
    ssim_vals = {7: [], 11: [], 17: [], 23: [], 31: []}
    global_ncc_vals = []
    windowed_ncc_vals = {7: [], 11: [], 17: [], 23: [], 31: []}

    gradient_l1_vals = []
    lpips_vals = []
    edge_pres_ratio_vals = []
    laplacian_blur_scores = []

    lpips_model = _build_lpips_model()

    for i in range(num_slices):
        if not mask[i]:
            continue
        g = gt_volume[i]
        p = pred_volume[i]

        l1_vals.append(np.mean(np.abs(p - g)))
        mie_vals.append(np.abs(p.mean() - g.mean()))

        mse = np.mean((p - g) ** 2)
        if mse == 0:
            psnr = float("inf")
        else:
            psnr = 10 * np.log10(1.0 / mse)
        psnr_vals.append(psnr)

        for w in ssim_vals.keys():
            try:
                ssim_val = skimage_ssim(g, p, data_range=1.0, win_size=w)
            except Exception:
                ssim_val = float("nan")
            ssim_vals[w].append(ssim_val)

        try:
            g_zero = g - g.mean()
            p_zero = p - p.mean()
            denom = np.sqrt(np.sum(g_zero**2) * np.sum(p_zero**2))
            if denom < 1e-8:
                ncc_val = 1.0 if np.allclose(g, p, atol=1e-6) else 0.0
            else:
                ncc_val = float(np.sum(g_zero * p_zero)) / denom
        except Exception:
            ncc_val = float("nan")
        global_ncc_vals.append(ncc_val)

        for w in windowed_ncc_vals.keys():
            try:
                local_ncc = compute_local_ncc(g, p, window_size=w)
                valid_vals = local_ncc[~np.isnan(local_ncc)].flatten()
                if len(valid_vals) == 0:
                    windowed_ncc_vals[w].append(float("nan"))
                else:
                    windowed_ncc_vals[w].append(np.median(valid_vals))
            except Exception:
                windowed_ncc_vals[w].append(float("nan"))

        g_grad = gradient_magnitude(g)
        p_grad = gradient_magnitude(p)
        gradient_l1_vals.append(np.mean(np.abs(g_grad - p_grad)))

        try:
            g_lpips = to_lpips_tensor(g)
            p_lpips = to_lpips_tensor(p)

            device = next(lpips_model.parameters()).device
            g_lpips = g_lpips.to(device)
            p_lpips = p_lpips.to(device)

            lpips_model.eval()
            with torch.no_grad():
                lpips_val = float(lpips_model(g_lpips, p_lpips).item())

        except Exception as e:
            log(f"LPIPS error at slice {i}: {e}")
            lpips_val = float("nan")

        lpips_vals.append(lpips_val)

        gt_edge = edge_strength(g)
        pred_edge = edge_strength(p)
        edge_ratio = pred_edge / (gt_edge + 1e-8)
        edge_pres_ratio_vals.append(edge_ratio)

        blur_score = laplacian_blur_score(p)
        blur_score_gt = laplacian_blur_score(g)
        laplacian_blur_scores.append(abs(blur_score - blur_score_gt))

    log("\nB-SCAN METRICS:")
    log(f" - L1 Error:                {float(np.mean(l1_vals)):.4f}")
    log(f" - MeanIntensityError:      {float(np.mean(mie_vals)):.4f}")
    log(f" - PSNR:                    {np.mean(psnr_vals):.4f}")
    log(f" - SSIM (win=11):           {np.nanmean(ssim_vals[11]):.4f}")
    log(f" - Global_NCC:              {np.nanmean(global_ncc_vals):.4f}")
    log(f" - Gradient L1:             {np.mean(gradient_l1_vals):.4f}")
    log(f" - LPIPS:                   {np.mean(lpips_vals):.4f}")
    log(f" - Edge Preservation Ratio: {np.mean(edge_pres_ratio_vals):.4f}")
    log(f" - Laplacian Blur Score:    {np.mean(laplacian_blur_scores):.4f}")

    return {
        "L1": np.mean(l1_vals),
        "MeanIntensityError": np.mean(mie_vals),
        "PSNR": np.mean(psnr_vals),
        "SSIM11": np.nanmean(ssim_vals[11]),
        "Global_NCC": np.nanmean(global_ncc_vals),
        "Gradient_L1": np.mean(gradient_l1_vals),
        "LPIPS": np.mean(lpips_vals),
        "Edge_Preservation_Ratio": np.mean(edge_pres_ratio_vals),
        "Laplacian_Blur_Score_Diff": np.mean(laplacian_blur_scores),
    }
