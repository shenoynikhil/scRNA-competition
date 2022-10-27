import pytorch_lightning as pl
import pickle
import torch
from scipy import sparse
import logging
from torch.utils.data import TensorDataset, DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(
        self, 
        x_path: str,
        y_path: str,
        x_test_path: str,
        splits_path: str = None,
        batch_size: int = 128,
    ):
        super().__init__()
        self.x_path = x_path
        self.y_path = y_path
        self.x_test_path = x_test_path
        self.splits_path = 'random' if splits_path is None else splits_path

        self.batch_size = batch_size

        self.setup()

    def setup(self, stage='fit'):
        if stage == 'fit' or stage is None:
            # Load Data
            logging.info("Loading data")
            x = pickle.load(open(self.x_path, "rb"))

            # load y as it is, since we need the original values to get metrics
            y = sparse.load_npz(self.y_path).toarray()

            # check splits
            if self.splits_path == 'random':
                n = x.shape[0]
                train_len = int(0.8 * n)
                x_train, x_val = torch.Tensor(x[:train_len, :]), torch.Tensor(x[train_len:, :])
                y_train, y_val = torch.Tensor(y[:train_len, :]), torch.Tensor(y[train_len:, :])
            else:
                raise NotImplementedError
            
            # create torch datasets
            self.train_dataset = TensorDataset(x_train, y_train)
            self.val_dataset = TensorDataset(x_val, y_val)
        
        elif stage == 'test' or stage is None:
            x_test = pickle.load(open(self.x_test_path, "rb"))

            self.test_dataset = TensorDataset(torch.Tensor(x_test))
        
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
        )
        
    def test_dataloader(self):
        return DataLoader(
            dataset = self.test_dataset,
            batch_size=self.batch_size,
        )
