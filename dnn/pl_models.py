from collections import defaultdict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

from .loss import CorrCoeffMSELoss


def corrcoeff(y_pred, y_true):
    """Pearson Correlation Coefficient
    Implementation in Torch, without shifting to cpu, detach, numpy (consumes time)
    """
    y_true_ = y_true - torch.mean(y_true, 1, keepdim=True)
    y_pred_ = y_pred - torch.mean(y_pred, 1, keepdim=True)

    num = (y_true_ * y_pred_).sum(1, keepdim=True)
    den = torch.sqrt(
        ((y_pred_**2).sum(1, keepdim=True)) * ((y_true_**2).sum(1, keepdim=True))
    )

    return (num / den).mean().item()


class BaseNet(pl.LightningModule):
    """Base Model using Simple NN without Context Conditioning"""

    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 100,
        num_layers: int = 5,
        layer_dim: int = 200,
        dropout: float = 0.2,
        activation: str = "ReLU",
        mse_weight: float = 1.0,
        pcc_weight: float = 0.0,
    ):
        """Initialization

        Parameters
        ----------
        input_dim: int
            Input dimension of input features
        output_dim: int
            Output dimension of target features
        num_layers: int
            Specifies how deep the neural network will be
        layer_dim: int
            Dimension of each inner layer
        dropout: float
            Specifies the `p` value of the dropout function
        activation: str
            Specifies which activation function to be used
            One of ['ReLU', 'SeLU', 'tanh']
        mse_weight: float
            Weight of MSE loss to be used
        pcc_weight: float
            Weight of PCC loss to be used
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layer_dim = layer_dim
        self.dropout = dropout
        self.activation = activation

        self.setup_net()
        self.loss = self.setup_loss(mse_weight, pcc_weight)

    def setup_pca(self, pca):
        self.pca = pca

    def on_fit_start(self) -> None:
        # store pcc values in each step
        self.pcc_storage = defaultdict(list)

    def setup_activation(self):
        """returns activation function based on activation"""
        if self.activation == "ReLU":
            return nn.ReLU()
        elif self.activation == "SeLU":
            return nn.SELU()
        elif self.activation == "tanh":
            return nn.Tanh()

        raise NotImplementedError

    def setup_net(self):
        """Setup Network"""
        modules = [
            nn.Linear(self.input_dim, self.layer_dim),
            self.setup_activation(),
            nn.Dropout(self.dropout),
        ]
        for _ in range(self.num_layers - 1):
            modules.append(nn.Linear(self.layer_dim, self.layer_dim))
            modules.append(self.setup_activation())
            modules.append(nn.Dropout(self.dropout))
        modules.append(nn.Linear(self.layer_dim, self.output_dim))
        self.net = nn.Sequential(*modules)

    def setup_loss(self, mse_weight: float, pcc_weight: float):
        return CorrCoeffMSELoss(mse_weight, pcc_weight)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return optim.Adam(self.net.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        """Training step"""
        return self.generic_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        return self.generic_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        """Test step"""
        return self.generic_step(batch, batch_idx, "test")

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        """Compute metrics at epoch level"""
        return self.generic_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        return self.generic_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        return self.generic_epoch_end(outputs, "test")

    def generic_step(self, batch, batch_idx, split):
        """Validation step"""
        # x, y
        x, y, y_orig = batch

        # get output of the network
        preds = self.net(x)

        # compute loss
        loss = self.loss(preds, y)

        return {
            "loss": loss,
            "preds": preds.detach().cpu(),
            "y": y.detach().cpu(),
            "y_orig": y_orig.detach().cpu(),
        }

    def generic_epoch_end(self, outputs: EPOCH_OUTPUT, split: str) -> None:
        """Compute PCC metric at epoch level and log to progressbar and update storage"""
        # raise to original dimension
        if self.pca is not None:
            epoch_preds = torch.cat(
                [pred["preds"].cpu() @ self.pca.components_ for pred in outputs]
            )
        else:
            epoch_preds = torch.cat([pred["preds"].cpu() for pred in outputs])
        epoch_y = torch.cat([pred["y_orig"].cpu() for pred in outputs])

        # compute pcc
        pcc = corrcoeff(epoch_preds, epoch_y)
        self.log(f"{split}/pcc", pcc, prog_bar=True, on_epoch=True, on_step=False)

        # update storage
        self.pcc_storage[split].append(pcc)


class ContextConditioningNet(BaseNet):
    """Extension of BaseNet using Conditioning"""

    def __init__(self, context_dim: int = 10, beta: float = 1e-3, **kwargs):
        self.context_dim = context_dim
        self.beta = beta
        super().__init__(**kwargs)

    def setup_net(self, hp):
        """Setup Network"""
        modules = [
            nn.Linear(self.input_dim, self.layer_dim),
            self.setup_activation(),
            nn.Dropout(self.dropout),
        ]
        for _ in range(self.num_layers - 1):
            modules.append(nn.Linear(self.layer_dim, self.layer_dim))
            modules.append(self.setup_activation())
            modules.append(nn.Dropout(self.dropout))
        modules.append(nn.Linear(self.layer_dim, self.output_dim))
        self.base_net = nn.Sequential(*modules)

        self.conditioning_layer = nn.Sequential(
            *[nn.Linear(self.context_dim, self.layer_dim), nn.ReLU()]
        )
        self.final_linear = nn.Linear(self.layer_dim, self.output_dim)
        self.net = nn.ModuleList(
            [
                self.base_net,
                self.conditioning_layer,
                self.final_linear,
            ]
        )

    def forward(self, x, z):
        """Final_Linear_Layer(Base(x) + Conditioning(Z))"""
        return self.net[2](self.net[0](x) + self.beta * self.net[1](z))

    def generic_step(self, batch, batch_idx, split):
        """Validation step"""
        # x, y
        x, y, y_orig = batch

        x, z = x[:, : -self.context_dim], x[:, -self.context_dim :]

        # get output of the network
        preds = self(x, z)

        # compute loss
        loss = self.loss(preds, y)

        return {
            "loss": loss,
            "preds": preds.detach(),
            "y": y.detach(),
            "y_orig": y_orig.detach(),
        }


class KaggleModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        (
            input_size,
            hidden_size,
            n_layers,
            output_size,
            activation,
            dropout,
            skip_connection,
        ) = (
            cfg["N_FEATURES"],
            cfg["HIDDEN_SIZE"],
            cfg["N_LAYERS"],
            cfg["N_TARGETS"],
            cfg["ACTIVATION"],
            cfg["DROPOUT"],
            cfg["SKIP_CONNECTION"],
        )

        self.skip_connection = skip_connection
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            activation(),
        )
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.LayerNorm(hidden_size),
                    activation(),
                )
                for _ in range(n_layers)
            ]
        )

        self.output = torch.nn.Sequential(
            *(
                [torch.nn.Dropout(0.1)]
                if dropout
                else []
                + [
                    torch.nn.Linear(hidden_size, output_size),
                    torch.nn.LayerNorm(output_size),
                    torch.nn.ReLU(),
                ]
            )
        )

    def forward(self, x):
        x = self.encoder(x)
        for block in self.blocks:
            if self.skip_connection:
                x = block(x) + x
            else:
                x = block(x)
        x = self.output(x)
        return x


class KaggleNet(BaseNet):
    def setup_net(self, hp):
        hp.update(
            {
                "ACTIVATION": torch.nn.SiLU,
            }
        )
        self.net = KaggleModel(hp)
