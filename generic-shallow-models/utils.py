import logging

import numpy as np
import scipy
from sklearn.decomposition import TruncatedSVD
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge


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


def preprocessing(config, x, y, x_test):
    if config["preprocessing"] == "TruncatedSVD":
        logging.info("Preprocessing Data using TruncatedSVD")

        # transform x and x_test
        pca_x = TruncatedSVD(
            n_components=config["preprocessing_params"]["input_dim"], random_state=config["seed"]
        )
        x_stacked = scipy.sparse.vstack([x, x_test])
        x_transformed = pca_x.fit_transform(x_stacked)

        x_train_transformed = x_transformed[: x.shape[0], :]
        x_test_transformed = x_transformed[x.shape[0] :, :]

        # delete arrays not going to be used from memory
        del x_transformed, x_stacked

        # transform y
        pca_y = TruncatedSVD(
            n_components=config["preprocessing_params"]["output_dim"], random_state=config["seed"]
        )
        y_transformed = pca_y.fit_transform(y)

        # return both y's transformed and (not) transformed
        return (
            pca_x,
            pca_y,
            x_train_transformed,
            y_transformed,
            y,
            x_test_transformed,
        )
    else:
        raise NotImplementedError


def setup_model(config):
    if config["model"] == "rbf_krr":
        logging.info("Setting up RBF based Kernel Regressor")
        return KernelRidge(
            alpha=config["model_params"]["alpha"], kernel=RBF(length_scale=config["model_params"]["scale"])
        )
    else:
        raise NotImplementedError
