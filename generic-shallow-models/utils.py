import logging

import numpy as np
from catboost import CatBoostRegressor, metrics
from lightgbm import LGBMRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.multioutput import MultiOutputRegressor
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
                objective=squared_log,
                random_state=config["seed"],
            )
        )
    elif config["model"] == "lgb":
        params = config["model_params"]
        logging.info(f"Setting up LightGBMRegressor: {params}")
        return MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=params.get("n_estimators", 100),
                objective=squared_log,
                random_state=config["seed"],
            )
        )
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
