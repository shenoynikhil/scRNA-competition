import logging
import pickle
from os.path import join
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset


class DataModule(pl.LightningDataModule):
    """Datamodule for all deep learning based models

    Parameters
    ----------
    x_path: str
        Path to loading the input data, could be in numpy or pickle format
    y_path: str
        Path to loading the target data, could be in numpy or pickle format
    output_dir: str
        Path to directory where artifacts are stored
    x_indices: str
        Basically a mapping of cell ids to the indices present in x. Used for splitting
        it into cross-validation and test splits
    cv_file: str
        Path to cell ids based cross validation splits, refer to `setup_splits()` to see
        how they are used to create cross validation splits
    eval_indices_path: str
        Path to the test set cell ids
    batch_size: int
        Batch size while creating dataloaders, default is 128
    preprocess_y: dict
        Contain information on how much the y dimension to be reduced using TruncatedSVD
    seed: int
        Seed for determinism, default is 42
    """

    def __init__(
        self,
        x_path: str,
        y_path: str,
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
        self.x_indices = x_indices
        self.eval_indices_path = eval_indices_path
        self.cv = "random" if cv_file is None else cv_file
        self.seed = seed
        self.preprocess_y = preprocess_y
        self.output_dir = output_dir

        self.batch_size = batch_size

        self.setup()

    def setup(self, stage="fit"):
        """Sets up the things needed to use this datamodule in the train stage.
        This is called in the init function. To get dataloader, post initializing the
        datamodule, just do,
        ```
        # some train indices
        indices = datamodule.splits[0][0]
        dl = datamodule.get_dataloader(indices)
        ```
        """
        if stage == "fit" or stage is None:
            # Load Data
            logging.info("Loading data")
            if ".pkl" in self.x_path:
                self.x = pickle.load(open(self.x_path, "rb"))
            elif ".npz" in self.x_path:
                self.x = sparse.load_npz(self.x_path).toarray()

            # load y as it is, since we need the original values to get metrics
            self.y = sparse.load_npz(self.y_path).toarray()

            # perform preprocessing if needed
            if self.preprocess_y:
                self.y_transformed, self.y, self.pca = self.perform_preprocessing(
                    self.y
                )
            else:
                self.y_transformed = self.y
                self.pca = None

            self.splits = self.setup_splits(stage="fit")

    def setup_splits(self, stage: str = "fit"):
        """Returns either of the following,
        - Cross Validation Splits if in `fit` stage
        - Test Indices if in `test` stage
        """
        if stage == "fit":
            if self.cv == "random":
                # perform KFold cross validation
                logging.info("Setting up random split cross validation")
                np.random.seed(self.seed)
                all_row_indices = np.arange(self.x.shape[0])
                np.random.shuffle(all_row_indices)

                kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)

                return [
                    (tr_indices, val_indices)
                    for (tr_indices, val_indices) in kf.split(all_row_indices)
                ]
            else:
                logging.info("Performing CV based on splits provided")
                # also read indices data
                self.x_indices = np.load(self.x_indices, allow_pickle=True)[
                    "index"
                ].tolist()

                # perform cv
                with open(self.cv, "rb") as f:
                    cv_splits = pickle.load(f)

                return [
                    (
                        [i for i, x in enumerate(self.x_indices) if x in v["train"]],
                        [i for i, x in enumerate(self.x_indices) if x in v["val"]],
                    )
                    for v in cv_splits.values()
                ]
        elif stage == "test":
            if self.eval_indices_path:
                # get cell ids to be used as a test set
                self.eval_indices = np.load(
                    self.eval_indices_path, allow_pickle=True
                ).tolist()
                return [
                    i for i, x in enumerate(self.x_indices) if x in self.eval_indices
                ]
            else:
                return None
        else:
            raise NotImplementedError

    def perform_preprocessing(self, y):
        """Reduces dimension of y if `preprocess_y` dict has been provided"""
        pca_y = TruncatedSVD(
            n_components=self.preprocess_y["output_dim"],
            random_state=self.seed,
        )

        y_transformed = pca_y.fit_transform(y)
        filename = join(self.output_dir, f"pca_y.pkl")
        with open(filename, "wb") as file:
            pickle.dump(pca_y, file)
        return y_transformed, y, pca_y

    def get_dataset(self, indices):
        """Get dataset corresponding to indices"""
        return TensorDataset(
            torch.Tensor(self.x[indices, :]),
            torch.Tensor(self.y_transformed[indices, :]),
            torch.Tensor(self.y[indices, :]),
        )

    def get_dataloader(self, indices: Any):
        """Returns train_dataloader corresponding to particular split"""
        return DataLoader(
            dataset=self.get_dataset(indices),
            batch_size=self.batch_size,
        )
