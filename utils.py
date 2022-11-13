import logging

import numpy as np
import torch.optim as optim
from catboost import CatBoostRegressor, metrics, MultiTargetCustomMetric
from lightgbm import LGBMRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.multioutput import MultiOutputRegressor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from xgboost import XGBRegressor


def correlation_score(y_true, y_pred):
    """Scores the predictions according to the competition rules.

    It is assumed that the predictions are not constant.

    Returns the average of each sample's Pearson correlation coefficient

    Source: https://www.kaggle.com/code/xiafire/lb-t15-msci-multiome-catboostregressor#Predicting
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes are different.")
    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)


def setup_model(config, **kwargs):
    if config["model"] == "rbf_krr":
        params = config["model_params"]
        logging.info(f"Setting up RBF based Kernel Regressor: {params}")
        return KernelRidge(
            alpha=params["alpha"], kernel=RBF(length_scale=params["scale"])
        )
    elif config["model"] == "catboost":
        params = config["model_params"]
        logging.info(f"Setting up CatBoostRegressor: {params}")
        return CatBoostRegressor(
            iterations=params["iterations"],
            loss_function=metrics.MultiRMSE(),
            task_type=params.get("task_type", 'CPU'),
            random_seed=config["seed"],
            eval_metric=PCCCatBoostMetric(pca=kwargs.get('pca'))
        )
    elif config["model"] == "xgboost":
        params = config["model_params"]
        logging.info(f"Setting up XGBooxtRegressor: {params}")
        return MultiOutputRegressor(
            XGBRegressor(
                n_estimators=params.get("n_estimators", 100),
                objective="mae",
                random_state=config["seed"],
            )
        )
    elif config["model"] == "lgb":
        params = config.get("model_params", {})
        logging.info(f"Setting up LightGBMRegressor: {params}")
        return MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=params.get("n_estimators", 100),
                objective="mae",
                max_depth=params.get("max_depth", 8),
                learning_rate=params.get("learning_rate", 0.024),
                colsample_bytree=params.get("colsample_bytree", 0.564),
                subsample=params.get("subsample", 0.41),
                alpha=params.get("alpha", 1.136),
                lambda_l2=params.get("lambda_l2", 1.926e-05),
                min_child_weight=params.get("min_child_weight", 10.43),
                random_state=config["seed"],
            )
        )
    elif config["model"] == "tabnet":
        params = config.get("model_params", {})
        return TabNetRegressor(
            n_d=config.get('n_d', 16),
            n_a=config.get('n_a', 16),
            n_steps=config.get('n_steps', 3),
            lambda_sparse = config.get('lambda_sparse', 0),
            mask_type = config.get('mask_type', "entmax"),
            scheduler_params = config.get(
                'scheduler_params', 
                dict(
                    mode = "min", patience = 5, min_lr = 1e-5, factor = 0.9
                )
            ),
            device_name=config.get('device_name', 'cpu'),
            seed=config["seed"],
            verbose=1,
        )
    else:
        raise NotImplementedError

def pcc_lightgbm(dy_true, dy_pred):
    """An eval metric that always returns the same value"""
    metric_name = 'pcc'
    value = correlation_score(dy_true, dy_pred)
    is_higher_better = True
    return metric_name, value, is_higher_better


def gradient(predt: np.ndarray, dtrain) -> np.ndarray:
    """Compute the gradient squared error."""
    y = dtrain.reshape(predt.shape)
    return (predt - y).reshape(y.size)


def hessian(predt: np.ndarray, dtrain) -> np.ndarray:
    """Compute the hessian for squared error."""
    return np.ones(predt.shape).reshape(predt.size)


def squared_log(predt, dtrain):
    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)
    return grad, hess


def get_hypopt_space(model_type: str, trial, seed: int = 42):
    """Returns parameters for model for conducting hyperoptimization

    Currently supports one of ['lgbm']

    Parameters
    ----------
    model_type: str
        Type of model
    trial:
        <insert>
    seed: int
        Random seed to be set

    Returns
    -------
    params: dict
        dictionary of parameters, could be passed to model as
        `model(**params)`
    """
    if model_type == "lgb":
        return {
            "verbosity": 1,  # 0 (silent) - 3 (debug)
            "objective": "rmse",
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, 50),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.05, log=True
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.2, 0.6, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.4, 0.8, log=True),
            "alpha": trial.suggest_float("alpha", 0.01, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 10, 1000, log=True
            ),
            "seed": seed,
            "n_jobs": 1,
        }
    elif model_type == 'tabnet':
        x = trial.suggest_categorical('n_x', [4, 8, 16, 24, 32, 40])
        return {
            "n_d": x,
            "n_a": x,
            "n_steps": trial.suggest_int('n_steps', 3, 10, 1),
        }
    elif model_type == 'smartNN':
        num_layers = trial.suggest_int('num_layers', 3, 6)
        layers = []
        for i in range(num_layers):
            n_units = int(trial.suggest_int(f'n_units_{i}', 128, 1029))
            layers.append(n_units)
        dropout = trial.suggest_uniform('dropout', 0.0, 0.8)
        return {
            'layers': layers,
            'dropout': dropout,
            'epochs': 50
        }
    else:
        raise NotImplementedError


class PCCCatBoostMetric(MultiTargetCustomMetric):
    def __init__(self, pca):
        self.pca = pca

    def is_max_optimal(self):
        # Returns whether great values of metric are better
        return True

    def evaluate(self, approxes, target, weight):
        # approxes is a list of indexed containers
        # (containers with only __len__ and __getitem__ defined),
        # one container per approx dimension.
        # Each container contains floats.
        # weight is a one dimensional indexed container.
        # target is a one dimensional indexed container.

        # weight parameter can be None.
        # Returns pair (error, weights sum)
        y_true, y_pred = (
            np.stack(target, 1) @ self.pca.components_, 
            np.stack(approxes, 1) @ self.pca.components_
        )
        corrsum = 0
        for i in range(len(y_true)):
            corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
        return corrsum, y_true.shape[0]

    def get_final_error(self, error, weight):
        # Returns final value of metric based on error and weight
        return error / weight
