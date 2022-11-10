from collections import defaultdict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

from .loss import CorrCoeffMSELoss


class BaseNet(pl.LightningModule):
    """Base Model using Simple NN without Context Conditioning"""

    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 100,
        hp: dict = {
            "layers": [170, 300, 480, 330, 770],
            "dropout": 0.2,
        },
        mse_weight: float = 1.0,
        pcc_weight: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.setup_net(hp)
        self.loss = self.setup_loss(mse_weight, pcc_weight)

    def setup_pca(self, pca):
        self.pca = pca

    def on_fit_start(self) -> None:
        # store pcc values in each step
        self.pcc_storage = defaultdict(list)

    def setup_net(self, hp):
        """Setup Network"""
        layer_shapes, dropout = hp["layers"], hp["dropout"]
        modules = [
            nn.Dropout(dropout),
            nn.Linear(self.input_dim, layer_shapes[0]),
            nn.ReLU(),
        ]
        for i in range(len(hp["layers"]) - 1):
            modules.append(nn.Linear(layer_shapes[i], layer_shapes[i + 1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(layer_shapes[-1], self.output_dim))
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

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        """Compute metrics at epoch level"""
        return self.generic_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        return self.generic_epoch_end(outputs, "val")

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
            "preds": preds.detach(),
            "y": y.detach(),
            "y_orig": y_orig.detach(),
        }

    def generic_epoch_end(self, outputs: EPOCH_OUTPUT, split: str) -> None:
        """Compute PCC metric at epoch level and log to progressbar and update storage"""
        # raise to original dimension
        if self.pca is not None:
            epoch_preds = torch.cat(
                [pred["preds"] @ self.pca.components_ for pred in outputs]
            )
        else:
            epoch_preds = torch.cat([pred["preds"] for pred in outputs])
        epoch_y = torch.cat([pred["y_orig"] for pred in outputs])

        # compute pcc
        pcc = corrcoeff(epoch_preds, epoch_y)
        self.log(f"{split}/pcc", pcc, prog_bar=True, on_epoch=True, on_step=False)

        # update storage
        self.pcc_storage[split].append(pcc)


class BasicDNN(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        hidden_size: int = 1024, 
        n_layers: int = 2,
        activation = torch.nn.SiLU, 
        dropout: bool = True,
        skip_connection: bool = True
    ):
        super().__init__()
        self.skip_connection = skip_connection
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            activation(),
        )
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.LayerNorm(hidden_size),
                activation(),
            )
            for _ in range(n_layers)]
        )

        self.output = torch.nn.Sequential(
            *(
                    [torch.nn.Dropout(0.1)] if dropout else [] +
                    [
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


class BabelMultiome(pl.LightningModule):
    def __init__(self, hp: dict):
        super().__init__()
        # setup network
        self.setup_net(hp)

    def setup_pca(self, pca):
        self.pca = pca

    def setup_net(self, hp):
        """Setup Network"""
        latent_dim = hp.get('latent_dim', 16)
        atac_dim = hp.get('atac_dim', 228942)
        rna_dim = hp.get('rna_dim', 22050)

        # contains, [rna_encoder, rna_decoder, atac_encoder, atac_decoder]
        self.net = nn.ModuleList([
            BasicDNN(rna_dim, latent_dim),
            BasicDNN(latent_dim, rna_dim),
            BasicDNN(atac_dim, latent_dim),
            BasicDNN(latent_dim, atac_dim),
        ])

    def configure_optimizers(self):
        return optim.Adam(
            self.net.parameters(), 
            lr=0.001
        )
    
    def forward(self, x):
        '''Get RNA from ATAC'''
        return self.net[1](self.net[2](x))
    
    def compute_loss(
        self, 
        rna_decoded_from_rna, 
        rna_decoded_from_atac,
        atac_decoded_from_atac, 
        atac_decoded_from_rna,
        rna,
        atac     
    ):
        '''Computes loss based on MSE'''
        return (
            F.mse_loss(rna_decoded_from_rna, rna) +
            F.mse_loss(rna_decoded_from_atac, rna) +
            F.mse_loss(atac_decoded_from_atac, atac) +
            F.mse_loss(atac_decoded_from_rna, atac)
        )
    
    def generic_step(self, batch, batch_idx, split):
        """Validation step"""
        # x (atac), y (rna)
        x, y, y_orig = batch

        # get latents using encoders
        rna_latent = self.net[0](y)
        atac_latent = self.net[2](x)
        
        # get decoded outputs
        rna_decoded_from_rna = self.net[1](rna_latent)
        rna_decoded_from_atac = self.net(1)(atac_latent)
        atac_decoded_from_atac = self.net[3](atac_latent)
        atac_decoded_from_rna = self.net[3](rna_latent)

        # compute loss based on reconstruction
        loss = self.compute_loss(
            rna_decoded_from_rna, 
            rna_decoded_from_atac,
            atac_decoded_from_atac, 
            atac_decoded_from_rna,
            y, # rna
            x # atac
        )

        return {
            "loss": loss,
            "preds": rna_decoded_from_atac.detach(),
            "y": y.detach(),
            "y_orig": y_orig.detach(),
        }
