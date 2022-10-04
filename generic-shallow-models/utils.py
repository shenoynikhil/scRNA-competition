import logging

import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge

from catboost import CatBoostRegressor, metrics


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
            iterations=params['iterations'],
            loss_function=metrics.MultiRMSE(), 
            random_seed=config['seed']
        )
    else:
        raise NotImplementedError
