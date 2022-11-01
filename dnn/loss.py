import torch
import torch.nn as nn


class CorrCoeffMSELoss(nn.Module):
    """Weighted Combination CorrelationCoeff Loss + MSE Loss"""

    def __init__(
        self,
        mse_weight: float = 1.0,
        pcc_weight: float = 1.0,
    ):
        """Initialization

        Parameters
        ----------
        mse_weight: float
            Weight of MSE Loss
        pcc_weight: float
            Weight of PCC loss
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.pcc_weight = pcc_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        """Forward pass"""
        y_true_ = y_true - torch.mean(y_true, 1, keepdim=True)
        y_pred_ = y_pred - torch.mean(y_pred, 1, keepdim=True)

        num = (y_true_ * y_pred_).sum(1, keepdim=True)
        den = torch.sqrt(
            ((y_pred_**2).sum(1, keepdim=True))
            * ((y_true_**2).sum(1, keepdim=True))
        )

        return self.pcc_weight * (
            1 - (num / den).mean()
        ) + self.mse_weight * self.mse_loss(y_pred, y_true)
