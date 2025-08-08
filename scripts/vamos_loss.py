import torch
import torch.nn as nn
import torch.nn.functional as F


class VAMOS_Loss(nn.Module):
    def __init__(
        self,
        lambda_weighted_mse=1.0,
        lambda_mip_axial=1.0,
        lambda_mip_coronal=1.0,
        lambda_aip_axial=1.0,
        lambda_aip_coronal=1.0,
        wl_alpha=100.0,
        wl_gamma=1/3,
        disable_wl_weighting=False
    ):
        """
        Composite hybrid loss for OCTA inpainting with structural and projection-aware terms.

        lambda_weighted_mse: Vessel-aware weighted MSE
        lambda_mip_axial: Maximum Intensity Projection (MIP) loss multiplier for axial axis
        lambda_mip_coronal: Maximum Intensity Projection (MIP) loss multiplier for coronal axis
        lambda_aip_axial: Average Intensity Projection (AIP) loss for axial axis
        lambda_aip_coronal: Average Intensity Projection (AIP) loss for coronal axis
        disable_wl_weighting: If True, disables vessel-weighting for the MSE term

        wl_alpha, wl_gamma: weight map shaping parameters for vessel-weighted term
        """
        super().__init__()
        self.lambda_weighted_mse = lambda_weighted_mse
        self.lambda_mip_axial = lambda_mip_axial
        self.lambda_mip_coronal = lambda_mip_coronal
        self.lambda_aip_axial = lambda_aip_axial
        self.lambda_aip_coronal = lambda_aip_coronal

        self.wl_alpha = wl_alpha
        self.wl_gamma = wl_gamma
        self.disable_wl_weighting = disable_wl_weighting


    def forward(self, pred, target):
        # pred, target: (B, 1, H, W)

        if not self.disable_wl_weighting:
            # Weighted pixel-wise MSE
            with torch.no_grad():
                eps = 1e-5
                pred_clamped = pred.detach().clamp(min=eps)
                target_clamped = target.clamp(min=eps)

                weight_map = (
                    self.wl_alpha * pred_clamped.pow(self.wl_gamma) +
                    target_clamped.pow(self.wl_gamma) +
                    0.5
                )
        else:
            weight_map = torch.ones_like(pred)

        mse_weighted = ((pred - target) ** 2 * weight_map).mean()

        # === Multi-Axis Orthogonal Projection Supervision ===

        # Axial: MIP over height (z→xy) → (B, 1, W)
        mip_axial_pred = pred.max(dim=2).values  # (B, 1, W)
        mip_axial_target = target.max(dim=2).values
        loss_mip_axial = F.l1_loss(mip_axial_pred, mip_axial_target)

        # Coronal: MIP over width (y→xz) → (B, 1, H)
        mip_coronal_pred = pred.max(dim=3).values  # (B, 1, H)
        mip_coronal_target = target.max(dim=3).values
        loss_mip_coronal = F.l1_loss(mip_coronal_pred, mip_coronal_target)

        # === AIP Losses ===
        # Axial AIP: mean over height (B, 1, W)
        aip_axial_pred = pred.mean(dim=2)
        aip_axial_target = target.mean(dim=2)
        loss_aip_axial = F.l1_loss(aip_axial_pred, aip_axial_target)

        # Coronal AIP: mean over width (B, 1, H)
        aip_coronal_pred = pred.mean(dim=3)
        aip_coronal_target = target.mean(dim=3)
        loss_aip_coronal = F.l1_loss(aip_coronal_pred, aip_coronal_target)


        total_loss = (
            self.lambda_weighted_mse * mse_weighted +
            self.lambda_mip_axial * loss_mip_axial +
            self.lambda_mip_coronal * loss_mip_coronal +
            self.lambda_aip_axial * loss_aip_axial +
            self.lambda_aip_coronal * loss_aip_coronal
        )

        return total_loss, {
            "weighted_mse": mse_weighted.item(),
            "mip_loss_axial": loss_mip_axial.item(),
            "mip_loss_coronal": loss_mip_coronal.item(),
            "aip_loss_axial": loss_aip_axial.item(),
            "aip_loss_coronal": loss_aip_coronal.item()
        }
