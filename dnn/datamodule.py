import numpy as np
import pytorch_lightning as pl
import pickle
import torch
from scipy import sparse
import logging
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        x_path: str,
        y_path: str,
        x_test_path: str,
        x_indices: str = None,
        cv_file: str = None,
        batch_size: int = 128,
        seed: int = 42,
    ):
        super().__init__()
        self.x_path = x_path
        self.y_path = y_path
        self.x_test_path = x_test_path
        self.x_indices = x_indices
        self.cv = 'random' if cv_file is None else cv_file
        self.seed = seed

        self.batch_size = batch_size

        self.setup()
    
    def setup_cv(self, x, y):
        self.train_datasets = []
        self.val_datasets = []        
        if self.cv == 'random':
            # perform KFold cross validation
            logging.info("Setting up random split cross validation")
            np.random.seed(self.seed)
            all_row_indices = np.arange(x.shape[0])
            np.random.shuffle(all_row_indices)

            kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)

            for i, (tr_indices, val_indices) in enumerate(kf.split(all_row_indices)):
                # preparing ith fold, for y_val we will use (not)transformed vector to calculate scores
                logging.info(f"{i}th fold")
                x_train, y_train = (
                    torch.Tensor(x[tr_indices, :]),
                    torch.Tensor(y[tr_indices, :]),
                )
                x_val, y_val = (torch.Tensor(x[val_indices, :]), torch.Tensor(y[val_indices, :]))
                self.train_datasets.append(TensorDataset(x_train, y_train))
                self.val_datasets.append(TensorDataset(x_val, y_val))
        else:
            logging.info('Performing CV based on splits provided') 
            # also read indices data
            self.x_indices = np.load(self.x_indices, allow_pickle=True)[
                "index"
            ].tolist()

            # perform cv
            with open(self.cv, "rb") as f:
                self.cv_splits = pickle.load(f)
            
            for _, (cv_split, split_dict) in enumerate(self.cv_splits.items()):
                # train ids and val ids to be used --> convert to set
                train_ids_set = set(split_dict["train"])
                val_ids_set = set(split_dict["val"])

                # get indices
                tr_indices, val_indices = (
                    [i for i, x in enumerate(self.x_indices) if x in train_ids_set],
                    [i for i, x in enumerate(self.x_indices) if x in val_ids_set],
                )
                
                # preparing ith fold, for y_val we will use (not)transformed vector to calculate scores
                logging.info(f"{cv_split}th fold")
                x_train, y_train = (
                    torch.Tensor(x[tr_indices, :]),
                    torch.Tensor(y[tr_indices, :]),
                )
                x_val, y_val = (torch.Tensor(x[val_indices, :]), torch.Tensor(y[val_indices, :]))
                self.train_datasets.append(TensorDataset(x_train, y_train))
                self.val_datasets.append(TensorDataset(x_val, y_val))


    def setup(self, stage='fit'):
        if stage == 'fit' or stage is None:
            # Load Data
            logging.info("Loading data")
            x = pickle.load(open(self.x_path, "rb"))

            # load y as it is, since we need the original values to get metrics
            y = sparse.load_npz(self.y_path).toarray()

            # check splits
            self.setup_cv(x, y)
        
        elif stage == 'test' or stage is None:
            x_test = pickle.load(open(self.x_test_path, "rb"))
            self.test_dataset = TensorDataset(torch.Tensor(x_test))
        
    def train_dataloader(self):
        return [
            DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
            )
            for dataset in self.train_datasets
        ]
    
    def val_dataloader(self):
        return [
            DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
            )
            for dataset in self.val_datasets
        ]
        
    def test_dataloader(self):
        return DataLoader(
            dataset = self.test_dataset,
            batch_size=self.batch_size,
        )
