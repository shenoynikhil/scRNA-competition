import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch
from collections import defaultdict
from pytorch_lightning.utilities.types import EPOCH_OUTPUT


def corrcoeff(y_pred, y_true):
    '''Pearson Correlation Coefficient
    Implementation in Torch, without shifting to cpu, detach, numpy (consumes time)
    '''
    y_true_ = y_true - torch.mean(y_true, 1, keepdim=True)
    y_pred_ = y_pred - torch.mean(y_pred, 1, keepdim=True)

    num = (y_true_ * y_pred_).sum(1, keepdim=True)
    den = torch.sqrt(((y_pred_ ** 2).sum(1, keepdim=True)) * ((y_true_ ** 2).sum(1, keepdim=True)))

    return ((num/den).mean().item())


class BaseNet(pl.LightningModule):
    '''Base Model using Simple NN without Context Conditioning'''
    def __init__(
        self,
        input_dim: int = 128, 
        output_dim: int = 100,
        hp: dict = {
            'layers': [170, 300, 480, 330, 770],
            'dropout': 0.2,
        }
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.setup_net(hp)
        self.loss = self.setup_loss()

    def on_fit_start(self) -> None:
        # store pcc values in each step
        self.pcc_storage = defaultdict(list)    
        
    def setup_net(self, hp):
        '''Setup Network'''
        layer_shapes, dropout = hp['layers'], hp['dropout']
        modules = [
            nn.Dropout(dropout),
            nn.Linear(self.input_dim, layer_shapes[0]),
            nn.ReLU(),
        ]
        for i in range(len(hp['layers']) - 1):
            modules.append(nn.Linear(layer_shapes[i], layer_shapes[i + 1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(layer_shapes[-1], self.output_dim))
        self.net = nn.Sequential(*modules)

    def setup_loss(self):
        return nn.MSELoss()
    
    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return optim.Adam(
            self.net.parameters(), lr=0.001
        )
    
    def training_step(self, batch, batch_idx):
        '''Training step'''
        return self.generic_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        '''Validation step'''
        return self.generic_step(batch, batch_idx, 'val')

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        '''Compute metrics at epoch level'''
        return self.generic_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        return self.generic_epoch_end(outputs, 'val')

    def generic_step(self, batch, batch_idx, split):
        '''Validation step'''
        # x, y
        x, y = batch

        # get output of the network
        preds = self.net(x)

        # compute loss
        loss = self.loss(preds, y)

        return {
            'loss': loss,
            'preds': preds.detach(),
            'y': y.detach()
        }

    def generic_epoch_end(self, outputs: EPOCH_OUTPUT, split: str) -> None:
        '''Compute PCC metric at epoch level and log to progressbar and update storage'''
        epoch_preds = torch.cat([pred['preds'] for pred in outputs])
        epoch_y = torch.cat([pred['y'] for pred in outputs])

        # compute pcc
        pcc = corrcoeff(epoch_preds, epoch_y)
        self.log(f'{split}/pcc', pcc, prog_bar=True, on_epoch=True, on_step=False)

        # update storage
        self.pcc_storage[split].append(pcc)


class ContextConditioningNet(BaseNet):
    '''Extension of BaseNet using Conditioning'''
    def __init__(
        self,
        context_dim: int = 10,
        input_dim: int = 128, 
        output_dim: int = 100,
        beta: float = 1e-3,
        hp: dict = {
            'layers': [170, 300, 480, 330, 770],
            'dropout': 0.2, 
        }
    ):
        self.context_dim = context_dim
        self.beta = beta
        super().__init__(input_dim, output_dim, hp)

    def setup_net(self, hp):
        '''Setup Network'''
        layer_shapes, dropout = hp['layers'], hp['dropout']
        modules = [
            nn.Dropout(dropout),
            nn.Linear(self.input_dim, layer_shapes[0]),
            nn.ReLU(),
        ]
        for i in range(len(hp['layers']) - 1):
            modules.append(nn.Linear(layer_shapes[i], layer_shapes[i + 1]))
            modules.append(nn.ReLU())
        self.final_linear = nn.Linear(layer_shapes[-1], self.output_dim)
        self.base_net = nn.Sequential(*modules)

        self.conditioning_layer = nn.Sequential(*[
            nn.Linear(self.context_dim, layer_shapes[-1]),
            nn.ReLU()
        ])
        self.net = nn.ModuleList([
            self.base_net,
            self.conditioning_layer,
            self.final_linear,
        ])

    def forward(self, x, z):
        '''Final_Linear_Layer(Base(x) + Conditioning(Z))'''
        return self.net[2](self.net[0](x) + self.beta * self.net[1](z))

    def generic_step(self, batch, batch_idx, split):
        '''Validation step'''
        # x, y
        x, y = batch

        x, z = x[:, :-self.context_dim], x[:, -self.context_dim:]

        # get output of the network
        preds = self(x, z)

        # compute loss
        loss = self.loss(preds, y)

        return {
            'loss': loss,
            'preds': preds.detach(),
            'y': y.detach()
        }
