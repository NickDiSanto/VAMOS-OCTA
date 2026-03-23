import torch
import torch.nn as nn
import torch.nn.functional as F


class VAMOS_Loss(nn.Module):
    """
    Composite loss for OCTA inpainting with vessel-weighted and projection-based terms.

    Args:
        lambda_weighted_mse: Weight for the vessel-weighted MSE term.
        lambda_mip_axial: Weight for axial maximum-intensity projection supervision.
        lambda_mip_lateral: Weight for lateral maximum-intensity projection supervision.
        lambda_aip_axial: Weight for axial average-intensity projection supervision.
        lambda_aip_lateral: Weight for lateral average-intensity projection supervision.
        wl_alpha: Scale factor for the vessel-weight map.
        wl_gamma: Exponent used in the vessel-weight map.
        disable_wl_weighting: If True, replace the vessel-weight map with ones.
    """

    def __init__(
        self,
        lambda_weighted_mse=1.0,
        lambda_mip_axial=1.0,
        lambda_mip_lateral=1.0,
        lambda_aip_axial=1.0,
        lambda_aip_lateral=1.0,
        wl_alpha=100.0,
        wl_gamma=1 / 3,
        disable_wl_weighting=False,
    ):
        super().__init__()
        self.lambda_weighted_mse = lambda_weighted_mse
        self.lambda_mip_axial = lambda_mip_axial
        self.lambda_mip_lateral = lambda_mip_lateral
        self.lambda_aip_axial = lambda_aip_axial
        self.lambda_aip_lateral = lambda_aip_lateral
        self.wl_alpha = wl_alpha
        self.wl_gamma = wl_gamma
        self.disable_wl_weighting = disable_wl_weighting

    def _build_weight_map(self, pred, target):
        if self.disable_wl_weighting:
            return torch.ones_like(pred)

        with torch.no_grad():
            eps = 1e-5
            pred_clamped = pred.detach().clamp(min=eps)
            target_clamped = target.clamp(min=eps)
            return (
                self.wl_alpha * pred_clamped.pow(self.wl_gamma)
                + target_clamped.pow(self.wl_gamma)
                + 0.5
            )

    def forward(self, pred, target):
        """Compute total loss and per-term diagnostics for a batch of center-slice predictions."""
        weight_map = self._build_weight_map(pred, target)
        mse_weighted = ((pred - target) ** 2 * weight_map).mean()

        # Projection terms supervise consistency in orthogonal views while keeping
        # the prediction target as a single reconstructed center slice.
        mip_axial_pred = pred.max(dim=2).values
        mip_axial_target = target.max(dim=2).values
        loss_mip_axial = F.l1_loss(mip_axial_pred, mip_axial_target)

        mip_lateral_pred = pred.max(dim=3).values
        mip_lateral_target = target.max(dim=3).values
        loss_mip_lateral = F.l1_loss(mip_lateral_pred, mip_lateral_target)

        aip_axial_pred = pred.mean(dim=2)
        aip_axial_target = target.mean(dim=2)
        loss_aip_axial = F.l1_loss(aip_axial_pred, aip_axial_target)

        aip_lateral_pred = pred.mean(dim=3)
        aip_lateral_target = target.mean(dim=3)
        loss_aip_lateral = F.l1_loss(aip_lateral_pred, aip_lateral_target)

        total_loss = (
            self.lambda_weighted_mse * mse_weighted
            + self.lambda_mip_axial * loss_mip_axial
            + self.lambda_mip_lateral * loss_mip_lateral
            + self.lambda_aip_axial * loss_aip_axial
            + self.lambda_aip_lateral * loss_aip_lateral
        )

        return total_loss, {
            "weighted_mse": mse_weighted.item(),
            "mip_loss_axial": loss_mip_axial.item(),
            "mip_loss_lateral": loss_mip_lateral.item(),
            "aip_loss_axial": loss_aip_axial.item(),
            "aip_loss_lateral": loss_aip_lateral.item(),
        }
