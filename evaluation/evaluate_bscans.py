import torch
import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim
import lpips

from evaluation.metrics_utils import compute_local_ncc, gradient_magnitude, laplacian_blur_score, edge_strength, to_lpips_tensor
from utils.logging import log


def evaluate_bscans(gt_volume, pred_volume, mask):
    assert gt_volume.shape == pred_volume.shape, "Volume shapes must match"

    D, _, _ = gt_volume.shape
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

    lpips_model = lpips.LPIPS(net='alex').cuda() if torch.cuda.is_available() else lpips.LPIPS(net='alex')


    for i in range(D):
        if not mask[i]:
            continue
        g = gt_volume[i]
        p = pred_volume[i]


        # from scipy.ndimage import gaussian_filter
        # # Edge-preserving smoothing
        # # g = gaussian_filter(g, sigma=1.0)
        # p = gaussian_filter(p, sigma=1.0)


        # L1 and Mean Intensity
        l1_vals.append(np.mean(np.abs(p - g)))
        mie_vals.append(np.abs(p.mean() - g.mean()))

        # PSNR
        mse = np.mean((p - g) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 10 * np.log10(1.0 / mse)
        psnr_vals.append(psnr)

        # SSIM at multiple windows
        for w in ssim_vals.keys():
            try:
                ssim_val = skimage_ssim(g, p, data_range=1.0, win_size=w)
            except Exception:
                ssim_val = float('nan')
            ssim_vals[w].append(ssim_val)

        # Global NCC (slice-wide Pearson)
        try:
            g_zero = g - g.mean()
            p_zero = p - p.mean()
            denom = np.sqrt(np.sum(g_zero**2) * np.sum(p_zero**2))
            if denom < 1e-8:
                ncc_val = 1.0 if np.allclose(g, p, atol=1e-6) else 0.0
            else:
                ncc_val = float(np.sum(g_zero * p_zero)) / denom
        except Exception:
            ncc_val = float('nan')
        global_ncc_vals.append(ncc_val)

        # Windowed NCC
        for w in windowed_ncc_vals.keys():
            try:
                local_ncc = compute_local_ncc(g, p, window_size=w)
                valid_vals = local_ncc[~np.isnan(local_ncc)].flatten()
                if len(valid_vals) == 0:
                    windowed_ncc_vals[w].append(float('nan'))
                else:
                    windowed_ncc_vals[w].append(np.median(valid_vals))
            except Exception:
                windowed_ncc_vals[w].append(float('nan'))

        # Gradient L1
        g_grad = gradient_magnitude(g)
        p_grad = gradient_magnitude(p)
        gradient_l1_vals.append(np.mean(np.abs(g_grad - p_grad)))

        # LPIPS
        try:
            g_lpips = to_lpips_tensor(g)
            p_lpips = to_lpips_tensor(p)

            # Move to same device as model
            device = next(lpips_model.parameters()).device
            g_lpips = g_lpips.to(device)
            p_lpips = p_lpips.to(device)

            # Ensure model is in eval mode
            lpips_model.eval()
            with torch.no_grad():
                lpips_val = float(lpips_model(g_lpips, p_lpips).item())

        except Exception as e:
            log(f"LPIPS error at slice {i}: {e}")
            lpips_val = float('nan')

        lpips_vals.append(lpips_val)

        # Edge Preservation Ratio
        gt_edge = edge_strength(g)
        pred_edge = edge_strength(p)
        edge_ratio = pred_edge / (gt_edge + 1e-8)
        edge_pres_ratio_vals.append(edge_ratio)

        # Laplacian Blur Score (sharpness)
        blur_score = laplacian_blur_score(p)
        blur_score_gt = laplacian_blur_score(g)
        laplacian_blur_scores.append(abs(blur_score - blur_score_gt))

    # # 3D NCC
    # try:
    #     window_size_3d = (7, 7, 3)  # You can adjust this
    #     ncc3d_map = compute_local_3d_ncc(gt_volume, pred_volume, window_size=window_size_3d)
    #     # Only evaluate over corrupted slices
    #     valid_vals = ncc3d_map[mask]
    #     valid_vals = valid_vals[~np.isnan(valid_vals)]
    #     if len(valid_vals) > 0:
    #         ncc3d_mean = float(np.mean(valid_vals))
    #     else:
    #         ncc3d_mean = None
    # except Exception:
    #     ncc3d_mean = None

    log("\B-SCAN METRICS:")
    log(f" - L1 Error:                {float(np.mean(l1_vals)):.4f}")
    log(f" - MeanIntensityError:      {float(np.mean(mie_vals)):.4f}")
    log(f" - PSNR:                    {np.mean(psnr_vals):.4f}")
    # for w in [7, 11, 17, 23, 31]:
    #     log(f" - SSIM (win={w}):          {np.nanmean(ssim_vals[w]):.4f}")
    log(f" - SSIM (win=11):           {np.nanmean(ssim_vals[11]):.4f}")
    log(f" - Global_NCC:              {np.nanmean(global_ncc_vals):.4f}")
    # for w in [7, 11, 17, 23, 31]:
    #     log(f" - Windowed NCC (win={w}):  {metrics['Windowed_NCC'][w]}")
    # log(f" - Local 3D NCC (win={7}):    {ncc3d_mean:.4f}")
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
        "Laplacian_Blur_Score_Diff": np.mean(laplacian_blur_scores)
    }
