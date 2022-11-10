import logging
import pickle
from os.path import join

import numpy as np
import pytorch_lightning as pl
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        x_path: str,
        y_path: str,
        x_test_path: str,
        output_dir: str,
        x_indices: str = None,
        cv_file: str = None,
        eval_indices_path: str = None,
        batch_size: int = 128,
        preprocess_y: dict = None,
        seed: int = 42,
    ):
        super().__init__()
        self.x_path = x_path
        self.y_path = y_path
        self.x_test_path = x_test_path
        self.x_indices = x_indices
        self.eval_indices_path = eval_indices_path
        self.cv = "random" if cv_file is None else cv_file
        self.seed = seed
        self.preprocess_y = preprocess_y
        self.output_dir = output_dir

        self.batch_size = batch_size

        self.setup()

    def perform_preprocessing(self, y):
        pca_y = TruncatedSVD(
            n_components=self.preprocess_y["output_dim"],
            random_state=self.seed,
        )

        y_transformed = pca_y.fit_transform(y)
        filename = join(self.output_dir, f"pca_y.pkl")
        with open(filename, "wb") as file:
            pickle.dump(pca_y, file)
        return y_transformed, y, pca_y

    def setup_cv(self, x, y, y_orig):
        self.train_datasets = []
        self.val_datasets = []
        if self.cv == "random":
            # perform KFold cross validation
            logging.info("Setting up random split cross validation")
            np.random.seed(self.seed)
            all_row_indices = np.arange(x.shape[0])
            np.random.shuffle(all_row_indices)

            kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)

            for i, (tr_indices, val_indices) in enumerate(kf.split(all_row_indices)):
                # preparing ith fold, for y_val we will use (not)transformed vector to calculate scores
                logging.info(f"{i}th fold")
                x_train, y_train, y_orig_train = (
                    torch.Tensor(x[tr_indices, :]),
                    torch.Tensor(y[tr_indices, :]),
                    torch.Tensor(y_orig[tr_indices, :]),
                )
                x_val, y_val, y_orig_val = (
                    torch.Tensor(x[val_indices, :]),
                    torch.Tensor(y[val_indices, :]),
                    torch.Tensor(y_orig[val_indices, :]),
                )
                self.train_datasets.append(
                    TensorDataset(x_train, y_train, y_orig_train)
                )
                self.val_datasets.append(TensorDataset(x_val, y_val, y_orig_val))
        else:
            logging.info("Performing CV based on splits provided")
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
                x_train, y_train, y_orig_train = (
                    torch.Tensor(x[tr_indices, :]),
                    torch.Tensor(y[tr_indices, :]),
                    torch.Tensor(y_orig[tr_indices, :]),
                )
                x_val, y_val, y_orig_val = (
                    torch.Tensor(x[val_indices, :]),
                    torch.Tensor(y[val_indices, :]),
                    torch.Tensor(y_orig[val_indices, :]),
                )
                self.train_datasets.append(
                    TensorDataset(x_train, y_train, y_orig_train)
                )
                self.val_datasets.append(TensorDataset(x_val, y_val, y_orig_val))
        del x_train, y_train, y_orig_train, x_val, y_val, y_orig_val

    def setup(self, stage="fit"):
        if stage == "fit" or stage is None:
            # Load Data
            logging.info("Loading data")
            if '.pkl' in self.x_path:
                x = pickle.load(open(self.x_path, "rb"))
            elif '.npz' in self.x_path:
                x = sparse.load_npz(self.x_path).toarray()

            # load y as it is, since we need the original values to get metrics
            y = sparse.load_npz(self.y_path).toarray()

            # perform preprocessing if needed
            if self.preprocess_y:
                y_transformed, y, self.pca = self.perform_preprocessing(y)
            else:
                y_transformed = y
                self.pca = None

            # check splits
            self.setup_cv(x, y_transformed, y)
            del x, y, y_transformed
            
            # setup evaluation set
            if self.cv != 'random' and self.eval_indices_path is not None:
                logging.info("Setting up evaluation test set")
                eval_ids = np.load(self.eval_indices_path)
                eval_indices = [i for i, id_ in enumerate(self.x_indices) if id_ in eval_ids]

                # construct tensors
                x_eval, y_eval, y_orig_eval = (
                    torch.Tensor(x[eval_indices, :]),
                    torch.Tensor(y_transformed[eval_indices, :]),
                    torch.Tensor(y[eval_indices, :]),
                )
                self.eval_dataset = TensorDataset(x_eval, y_eval, y_orig_eval)

        elif stage == "test" or stage is None:
            logging.info("loading test data")
            if '.pkl' in self.x_test_path:
                x_test = pickle.load(open(self.x_test_path, "rb"))
            elif '.npz' in self.x_test_path:
                x_test = sparse.load_npz(self.x_test_path).toarray()

            # store as dataset
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
            dataset=self.test_dataset,
            batch_size=self.batch_size,
        )
    
    def eval_dataloader(self):
        return DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.batch_size,
        )
