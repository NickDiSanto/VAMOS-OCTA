import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim

from evaluation.metrics_utils import compute_psnr, gradient_magnitude
from utils.logging import log


def evaluate_projection(gt_volume, pred_volume):
    """Evaluate projection-space quality on axial max-intensity projections."""
    max_gt = np.amax(gt_volume, axis=1)
    max_pred = np.amax(pred_volume, axis=1)

    proj_l1 = np.mean(np.abs(max_pred - max_gt))
    proj_mie = abs(max_gt.mean() - max_pred.mean())
    proj_psnr = compute_psnr(max_gt, max_pred)
    proj_ssim = skimage_ssim(max_pred, max_gt, data_range=1.0, win_size=11)

    g_z = max_gt - max_gt.mean()
    p_z = max_pred - max_pred.mean()
    denom = np.sqrt(np.sum(g_z**2) * np.sum(p_z**2))
    proj_global_ncc = np.sum(g_z * p_z) / denom if denom > 1e-8 else 0.0

    grad_gt = gradient_magnitude(max_gt)
    grad_pred = gradient_magnitude(max_pred)
    proj_gradient_l1 = np.mean(np.abs(grad_gt - grad_pred))

    log("\nPROJECTION METRICS:")
    log(f" - Projection L1:           {proj_l1:.4f}")
    log(f" - Projection MIE:          {proj_mie:.4f}")
    log(f" - Projection PSNR:         {proj_psnr:.4f}")
    log(f" - Projection SSIM11:       {proj_ssim:.4f}")
    log(f" - Projection Global NCC:   {proj_global_ncc:.4f}")
    log(f" - Projection Gradient L1:  {proj_gradient_l1:.4f}")

    return {
        "Projection_L1": proj_l1,
        "Projection_MIE": proj_mie,
        "Projection_PSNR": proj_psnr,
        "Projection_SSIM11": proj_ssim,
        "Projection_Global_NCC": proj_global_ncc,
        "Projection_Gradient_L1": proj_gradient_l1,
    }
