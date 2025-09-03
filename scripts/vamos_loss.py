import torch
import torch.nn as nn
import torch.nn.functional as F


class VAMOS_Loss(nn.Module):
    def __init__(
        self,
        lambda_weighted_mse=1.0,
        lambda_mip_axial=1.0,
        lambda_mip_lateral=1.0,
        lambda_aip_axial=1.0,
        lambda_aip_lateral=1.0,
        wl_alpha=100.0,
        wl_gamma=1/3,
        disable_wl_weighting=False
    ):
        """
        Composite hybrid loss for OCTA inpainting with structural and projection-aware terms.

        lambda_weighted_mse: Vessel-aware weighted MSE
        lambda_mip_axial: Maximum Intensity Projection (MIP) loss multiplier for axial axis
        lambda_mip_lateral: Maximum Intensity Projection (MIP) loss multiplier for lateral axis
        lambda_aip_axial: Average Intensity Projection (AIP) loss for axial axis
        lambda_aip_lateral: Average Intensity Projection (AIP) loss for lateral axis
        disable_wl_weighting: If True, disables vessel-weighting for the MSE term

        wl_alpha, wl_gamma: weight map shaping parameters for vessel-weighted term
        """
        super().__init__()
        self.lambda_weighted_mse = lambda_weighted_mse
        self.lambda_mip_axial = lambda_mip_axial
        self.lambda_mip_lateral = lambda_mip_lateral
        self.lambda_aip_axial = lambda_aip_axial
        self.lambda_aip_lateral = lambda_aip_lateral

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

        # --- Multi-Axis Orthogonal Projection Supervision

        # Axial MIP: max over height (z→xy) → (B, 1, W)
        mip_axial_pred = pred.max(dim=2).values  # (B, 1, W)
        mip_axial_target = target.max(dim=2).values
        loss_mip_axial = F.l1_loss(mip_axial_pred, mip_axial_target)

        # Lateral MIP: max over width (y→xz) → (B, 1, H)
        mip_lateral_pred = pred.max(dim=3).values  # (B, 1, H)
        mip_lateral_target = target.max(dim=3).values
        loss_mip_lateral = F.l1_loss(mip_lateral_pred, mip_lateral_target)

        # Axial AIP: mean over height (B, 1, W)
        aip_axial_pred = pred.mean(dim=2)
        aip_axial_target = target.mean(dim=2)
        loss_aip_axial = F.l1_loss(aip_axial_pred, aip_axial_target)

        # Lateral AIP: mean over width (B, 1, H)
        aip_lateral_pred = pred.mean(dim=3)
        aip_lateral_target = target.mean(dim=3)
        loss_aip_lateral = F.l1_loss(aip_lateral_pred, aip_lateral_target)


        total_loss = (
            self.lambda_weighted_mse * mse_weighted +
            self.lambda_mip_axial * loss_mip_axial +
            self.lambda_mip_lateral * loss_mip_lateral +
            self.lambda_aip_axial * loss_aip_axial +
            self.lambda_aip_lateral * loss_aip_lateral
        )

        return total_loss, {
            "weighted_mse": mse_weighted.item(),
            "mip_loss_axial": loss_mip_axial.item(),
            "mip_loss_lateral": loss_mip_lateral.item(),
            "aip_loss_axial": loss_aip_axial.item(),
            "aip_loss_lateral": loss_aip_lateral.item()
        }
