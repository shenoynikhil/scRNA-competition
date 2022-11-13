import torch
import torch.nn as nn


class AsymmetricMSELoss(nn.Module):
    def forward(self, y_pred, y_true):
        '''Compute MSE loss where non-zero values of y_true has a higher weight
        (no_of_zero_elements / total_elements)

        Arguments
        ---------
        y_pred: torch.Tensor
            prediction tensor of shape [batch_size, output_dim]
        y_true: torch.Tensor
            target tensor of shape [batch_size, output_dim]
        '''
        # compute positive weight per item (no_of_pos)
        d = y_true.shape[1]
        pos_weight_per_item = (((y_true == 0.0) * 1.0).sum(1) / d).repeat(d, 1).T
        neg_weight_per_item = (1 - pos_weight_per_item)

        # mask
        mask = torch.where(y_true == 0.0, neg_weight_per_item,pos_weight_per_item)

        # compute the mse_loss
        mse_loss = (y_true - y_pred) ** 2
        # compute the masked_mse_loss
        masked_mse_loss = mask * mse_loss
        # compute average over dim = 1 and then over batch elements
        loss = masked_mse_loss.mean(1).mean()
        return loss


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
