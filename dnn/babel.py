'''Babel Based Networks for Cross Modality Prediction'''
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .pl_models import BaseNet
from .loss import AsymmetricMSELoss


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
                    ([torch.nn.Dropout(0.1)] if dropout else []) +
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


class Babel(BaseNet):
    def __init__(self, hp: dict):
        pl.LightningModule.__init__(self)
        # setup network
        self.hp = hp
        self.setup_net(hp)

    def setup_pca(self, pca):
        self.pca = pca

    def setup_net(self, hp):
        """Setup Network"""
        latent_dim = hp.get('latent_dim', 16)
        x_dim = hp.get('x_dim', 228942)
        y_dim = hp.get('y_dim', 22050)

        # contains, [x_encoder, x_decoder, y_encoder, y_decoder]
        n_layers = hp.get('n_layers', 2)
        self.net = nn.ModuleList([
            BasicDNN(x_dim, latent_dim, n_layers=n_layers),
            BasicDNN(latent_dim, x_dim, n_layers=n_layers),
            BasicDNN(y_dim, latent_dim, n_layers=n_layers),
            BasicDNN(latent_dim, y_dim, n_layers=n_layers),
        ])

        # check loss type
        loss_type = hp.get('loss', 'mse')
        if loss_type == 'mse':
            self.loss = nn.MSELoss()
        elif loss_type == 'AsymmetricMSELoss':
            self.loss = AsymmetricMSELoss()

    def configure_optimizers(self):
        return optim.Adam(
            self.net.parameters(), 
            lr=0.001
        )
    
    def forward(self, x):
        '''Get RNA from ATAC'''
        return self.net[3](self.net[0](x))
    
    def compute_loss(
        self, 
        x_from_x, 
        x_from_y,
        y_from_x, 
        y_from_y,
        x,
        y     
    ):
        '''Computes loss based on MSE'''
        return (
            self.loss(x_from_x, x) +
            self.loss(x_from_y, x) +
            self.loss(y_from_x, y) +
            self.loss(y_from_y, y)
        )
    
    def generic_step(self, batch, batch_idx, split):
        """Validation step"""
        # x (atac), y (rna)
        x, y, y_orig = batch

        # get outputs
        x_from_x = self.net[1](self.net[0](x)) 
        x_from_y = self.net[1](self.net[2](y))
        y_from_x = self.net[3](self.net[0](x)) 
        y_from_y = self.net[3](self.net[2](y))

        # compute loss based on reconstruction
        loss = self.compute_loss(
            x_from_x, 
            x_from_y,
            y_from_x, 
            y_from_y,
            x,
            y
        )

        return {
            "loss": loss,
            "preds": y_from_x.detach(),
            "y": y.detach(),
            "y_orig": y_orig.detach(),
        }
