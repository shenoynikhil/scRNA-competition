import logging

import numpy as np
import torch.optim as optim
from catboost import CatBoostRegressor, metrics
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


def setup_model(config):
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
            random_seed=config["seed"],
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
                random_state=config["seed"],
            )
        )
    elif config["model"] == "tabnet":
        params = config.get("model_params", {})
        tabnet_params = dict(
            n_d=16,
            n_a=16,
            n_steps=8,
            gamma=1.3,
            lambda_sparse=0,
            optimizer_fn=optim.Adam,
            optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
            mask_type="entmax",
            scheduler_params=dict(mode="min", patience=5, min_lr=1e-5, factor=0.9),
            scheduler_fn=ReduceLROnPlateau,
            seed=config["seed"],
            verbose=1,
        )
        return TabNetRegressor(**tabnet_params)
    else:
        raise NotImplementedError


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
    if model_type == "lgbm":
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
    else:
        raise NotImplementedError
