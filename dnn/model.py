import logging
import os
import pickle

import numpy as np
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         TQDMProgressBar)

from .babel import Babel
from .datamodule import DataModule
from .pl_models import BaseNet, ContextConditioningNet, KaggleNet


class DNNSetup:
    """Setup for training deep learning networks setup using Pytorch Lightning. This interface basically
    helps call
    ```python
    trainer = Trainer()
    model = Model()
    datamodule = DataModule()

    trainer.fit(model, datamodule)
    ```
    """

    def __init__(self, config):
        self.config = config
        self.output_dir = self.config["output_dir"]

    def setup_model(self, model_config: dict):
        model_type = model_config.get("model_type", "BaseNet")
        if model_type == "BaseNet":
            return BaseNet(
                input_dim=model_config.get("input_dim", 128),
                output_dim=model_config.get("output_dim", 100),
                layer_dim=model_config.get("layer_dim", 300),
                num_layers=model_config.get("num_layers", 4),
                dropout=model_config.get("dropout", 0.2),
                activation=model_config.get("activation", "ReLU"),
                mse_weight=model_config.get("mse_weight", 1.0),
                pcc_weight=model_config.get("pcc_weight", 0.0),
            )
        elif model_type == "ContextConditioningNet":
            return ContextConditioningNet(
                input_dim=model_config.get("input_dim", 128),
                output_dim=model_config.get("output_dim", 100),
                layer_dim=model_config.get("layer_dim", 300),
                dropout=model_config.get("dropout", 0.2),
                num_layers=model_config.get("num_layers", 4),
                activation=model_config.get("activation", "ReLU"),
                mse_weight=model_config.get("mse_weight", 1.0),
                pcc_weight=model_config.get("pcc_weight", 0.0),
                beta=model_config.get("beta", 1e-3),
                context_dim=model_config.get("context_dim", 10),
            )
        elif model_type == "KaggleNet":
            return KaggleNet(hp=model_config.get("hp"))
        elif model_type == "Babel":
            return Babel(hp=model_config.get("hp"))
        else:
            return NotImplementedError

    def setup_trainer(
        self, trainer_config: dict, split: int, save_checkpoints: bool = True
    ):
        """Setup trainer for experiments"""
        params = {
            "accelerator": "gpu",
            "devices": 1,
            "default_root_dir": os.path.join(self.output_dir, f"cv_{split}"),
            "logger": False,
            "num_sanity_val_steps": trainer_config.get("num_sanity_val_steps", 0),
            "max_epochs": trainer_config.get("max_epochs", 200),
            "callbacks": [
                TQDMProgressBar(refresh_rate=1000),
                EarlyStopping(monitor="val/pcc", mode="max", patience=20),
            ],
        }
        if save_checkpoints:
            params["callbacks"].append(
                ModelCheckpoint(filename="{epoch:02d}", monitor="val/pcc", mode="max")
            )
        return pl.Trainer(**params)

    def setup_datamodule(self, datamodule_config: dict):
        return DataModule(
            x_path=datamodule_config.get("x"),
            y_path=datamodule_config.get("y"),
            x_indices=datamodule_config.get("x_indices", None),
            cv_file=datamodule_config.get("cv_file", None),
            eval_indices_path=datamodule_config.get("eval_indices_path", None),
            output_dir=self.output_dir,
            preprocess_y=datamodule_config.get("preprocess_y", None),
            normalize_y=datamodule_config.get("normalize_y", True),
            batch_size=datamodule_config.get("batch_size", 128),
            seed=self.config.get("seed", 42),
        )

    def run_experiment(self):
        """Performs the experiment on different cv splits"""
        # fit the model on different cv splits
        datamodule = self.setup_datamodule(self.config.get("datamodule_config"))
        pca_y = datamodule.pca
        splits = datamodule.splits

        # get test set data
        test_splits = datamodule.setup_splits(stage="test")
        if test_splits:
            test_dl = datamodule.get_dataloader(test_splits)
            test_scores = []

        scores = []
        for i, (tr_indices, val_indices) in enumerate(splits):
            # create dataloaders based on indices
            tr_dl, vl_dl = (
                datamodule.get_dataloader(tr_indices),
                datamodule.get_dataloader(val_indices),
            )

            model = self.setup_model(self.config.get("model_config", {}))
            model.setup_pca(pca_y)
            trainer = self.setup_trainer(self.config.get("trainer_config", {}), i)
            trainer.fit(model, tr_dl, vl_dl)

            # retrieve best val score early stopping callback
            score = [cb for cb in trainer.callbacks if isinstance(cb, EarlyStopping)][
                0
            ].best_score.item()
            scores.append(score)
            logging.info(f"Score for this split {i}: {score}")

            if test_splits:
                trainer.test(model, test_dl)
                test_scores.append(model.pcc_storage["test"])

            # save pcc_storage if pcc_storage created
            if hasattr(model, "pcc_storage"):
                with open(
                    os.path.join(self.output_dir, f"pcc_vals_{i}.pkl"), "wb"
                ) as file:
                    pickle.dump(model.pcc_storage, file)

            if self.config.get("stop_after_first_cv", False):
                break

        # log best scores
        logging.info(f"Scores across all splits : {scores}")

        if test_splits:
            logging.info(
                f"Test Scores: {test_scores}, mean test score: {np.mean(test_scores)}"
            )

        return np.mean(scores)

    def conduct_hpo(self, n_trials: int = 10):
        """Conduct Hyperparameter Optimization using Optuna"""
        datamodule = self.setup_datamodule(self.config.get("datamodule_config"))
        pca_y = datamodule.pca
        splits = datamodule.splits

        # define objective function
        def objective(trial, splits, pca):
            self.config = update_config(self.config, trial)

            scores = []
            for i, (tr_indices, val_indices) in enumerate(splits):
                # create dataloaders based on indices
                tr_dl, vl_dl = (
                    datamodule.get_dataloader(tr_indices),
                    datamodule.get_dataloader(val_indices),
                )

                # initialize model
                model = self.setup_model(self.config.get("model_config", {}))
                model.setup_pca(pca)
                trainer = self.setup_trainer(self.config.get("trainer_config", {}), i)
                # fit model
                trainer.fit(model, tr_dl, vl_dl)

                # retrieve best val score early stopping callback
                score = [
                    cb for cb in trainer.callbacks if isinstance(cb, EarlyStopping)
                ][0].best_score.item()
                scores.append(score)

            return np.mean(scores)

        # run hyperopt
        logging.info("Creating Optuna hyperopt strategy")
        study = optuna.create_study(study_name="hpo-run", direction="maximize")
        study.optimize(
            lambda trial: objective(trial, splits, pca_y),
            n_trials=n_trials,
            n_jobs=-1,
        )

        # save best results
        with open("study", "wb") as file:
            pickle.dump(study, file)


def update_config(model_config, trial):
    """Function for updating experiment config accordingly"""
    # create a copy of the config to be returned in the end
    model_type = model_config.get("model_type", "BaseNet")
    if model_type == "BaseNet":
        model_config.update(
            {
                "dropout": trial.suggest_float(
                    "dropout",
                    low=0.1,
                    high=0.9,
                    step=0.1,
                ),
                "activation": trial.suggest_categorical(
                    "activation", ["ReLU", "SeLU", "tanh"]
                ),
                "layer_dim": trial.suggest_int("layer_dim", low=50, high=500, step=50),
                "num_layers": trial.suggest_int("num_layers", low=2, high=5, step=1),
                "mse_weight": trial.suggest_categorical(
                    "mse_weight", [0.1 * x for x in range(11)]
                ),
                "pcc_weight": trial.suggest_categorical(
                    "mse_weight", [0.1 * x for x in range(11)]
                ),
            }
        )
    elif model_type == "ContextConditioningNet":
        model_config.update(
            {
                "dropout": trial.suggest_float(
                    "dropout",
                    low=0.1,
                    high=0.9,
                    step=0.1,
                ),
                "activation": trial.suggest_categorical(
                    "activation", ["ReLU", "SeLU", "tanh"]
                ),
                "layer_dim": trial.suggest_int("layer_dim", low=50, high=500, step=50),
                "num_layers": trial.suggest_int("num_layers", low=2, high=5, step=1),
                "mse_weight": trial.suggest_categorical(
                    "mse_weight", [0.1 * x for x in range(11)]
                ),
                "pcc_weight": trial.suggest_categorical(
                    "mse_weight", [0.1 * x for x in range(11)]
                ),
                "beta": trial.suggest_float(
                    "beta", low=0.0001, high=1.0, step=0.0001, log=True
                ),
            }
        )

    return model_config
