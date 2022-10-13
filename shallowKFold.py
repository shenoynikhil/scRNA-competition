import gc
import logging
import pickle
from os import makedirs
from os.path import dirname, join

import numpy as np
import optuna
import scipy
from lightgbm import LGBMRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor

from base import ExperimentHelper
from utils import correlation_score, get_hypopt_space


class ShallowModelKFold(ExperimentHelper):
    def read_data(self):
        # Load Data
        logging.info("Loading data")
        x_train_transformed = pickle.load(open(self.config["paths"]["x"], "rb"))
        x_test_transformed = pickle.load(open(self.config["paths"]["x_test"], "rb"))

        # load y as it is, since we need the original values to get metrics
        y = scipy.sparse.load_npz(self.config["paths"]["y"])
        return x_train_transformed, x_test_transformed, y

    def perform_preprocessing(self, y):
        pca_y = TruncatedSVD(
            n_components=self.config["preprocessing_params"]["output_dim"],
            random_state=self.seed,
        )

        y_transformed = pca_y.fit_transform(y)
        filename = join(self.config["output_dir"], f"pca_y.pkl")
        with open(filename, "wb") as file:
            pickle.dump(pca_y, file)
        return y_transformed, y, pca_y

    def fit_model(self, x, y, y_orig, x_test, model, pca_y):
        # perform KFold cross validation
        logging.info("Setting up cross validation")
        np.random.seed(self.seed)
        all_row_indices = np.arange(x.shape[0])
        np.random.shuffle(all_row_indices)

        kf = KFold(
            n_splits=self.config.get("folds", 5), shuffle=True, random_state=self.seed
        )

        # Save the model (do not save when config['save_models'] == False)
        save_models = self.config.get("save_models", True)

        # for cross val scores
        scores = []

        # for x_test predictions
        predictions = []
        for i, (tr_indices, val_indices) in enumerate(kf.split(all_row_indices)):
            # preparing ith fold, for y_val we will use (not)transformed vector to calculate scores
            logging.info(f"{i}th fold")
            x_train, y_train = (
                x[tr_indices, :],
                y[tr_indices, :],
            )
            x_val, y_val = (x[val_indices, :], y_orig[val_indices, :])

            # fit and then delete training splits
            logging.info(f"Fitting the model for {i}th Fold")
            model = self._fit_model(model, x_train, y_train, x_val, y[val_indices, :])
            del x_train, y_train

            # save models if save_models=True
            if save_models:
                pkl_filename = join(self.config["output_dir"], f"model_{i}th_fold.pkl")
                with open(pkl_filename, "wb") as file:
                    pickle.dump(model, file)

            logging.info("Predicting and calculating metrics")
            scores.append(
                correlation_score(y_val, model.predict(x_val) @ pca_y.components_)
            )

            del x_val, y_val
            gc.collect()

            # Perform test predictions with cross val model
            predictions.append(model.predict(x_test) @ pca_y.components_)

            # ---------- TODO: DELETE THE NEXT TWO LINES LATER ----------
            if i == self.config.get("folds", 4):
                break
            # ---------- TODO: DELETE THE ABOVE TWO LINES LATER ----------

        # Again garbage collection to reduce unnecessary memory usage
        gc.collect()

        logging.info(f"Scores on 5-fold CV: {scores}")

        # post processing final predictions before pickle saving them
        if self.config.get("prediction_agg", "") == "mean":
            # mean over cross val predictions
            prediction = np.mean(predictions, axis=0)
        else:
            raise NotImplementedError

        # save numpy array
        if self.config.get("save_test_predictions", True):
            pkl_filename = join(self.config["output_dir"], f"test_pred.pkl")
            logging.info(f"Saving Predictions to {pkl_filename}")
            makedirs(dirname(pkl_filename), exist_ok=True)
            with open(pkl_filename, "wb") as file:
                pickle.dump(prediction, file)

        # return CV score
        return np.mean(scores)

    def run_experiment(self):
        # get data
        x_train_transformed, x_test_transformed, y = self.read_data()

        # perform processing
        y_transformed, y, pca_y = self.perform_preprocessing(y)
        gc.collect()

        # get model
        model = self.setup_model()

        # run experiment
        score = self.fit_model(
            x_train_transformed,
            y_transformed,
            y.toarray(),
            x_test_transformed,
            model,
            pca_y,
        )

        return score

    def _fit_model(self, model, x_train, y_train, x_val=None, y_val=None):
        """Fit the model correctly"""
        if self.config["model"] == "tabnet":
            assert (x_val is not None) and (
                y_val is not None
            ), "one/both of x_val and y_val is/are None"
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_train, y_train), (x_val, y_val)],
                eval_name=["train", "valid"],
                eval_metric=["mae", "rmse", "mse"],
                max_epochs=50,
                patience=50,
                batch_size=1024,
                virtual_batch_size=128,
            )
        else:
            model.fit(x_train, y_train)

        return model

    def conduct_hpo(
        self,
        subset_size: int = 5000,
        n_trials: int = 10,
        train_subset_frac: float = 0.8,
    ):
        """Conducts HPO if supported

        Only call, experiment.conduct_hpo()
        """

        # get data
        x_train_transformed, _, y = self.read_data()

        # perform processing
        y_transformed, y, pca_y = self.perform_preprocessing(y)
        gc.collect()

        # reduce to size
        logging.info("Reducing size")
        x_train_transformed, y, y_transformed = (
            x_train_transformed[:subset_size, :],
            y[:subset_size, :],
            y_transformed[:subset_size, :],
        )

        def objective(
            trial,
            x,
            y,
            y_orig,
            pca_y,
        ):
            # get hyperopt parameters
            params = get_hypopt_space(self.config["model"], trial, self.seed)

            train_len = int(train_subset_frac * x.shape[0])  # let's do random for now
            x_train, y_train, x_val, y_val_orig = (
                x[:train_len, :],
                y[:train_len, :],
                x[train_len:, :],
                y_orig[train_len:, :],
            )
            gc.collect()

            model = MultiOutputRegressor(LGBMRegressor(**params))
            model.fit(x_train, y_train)

            score = correlation_score(
                y_val_orig, model.predict(x_val) @ pca_y.components_
            )
            logging.info(f"Score for trial: {trial} is, {score}")
            return score

        # run hyperopt
        logging.info("Running hyperopt")
        study = optuna.create_study(study_name="hpo-run", direction="maximize")
        study.optimize(
            lambda trial: objective(
                trial,
                x_train_transformed,
                y_transformed,
                y.toarray(),
                pca_y,
            ),
            n_trials=n_trials,
            n_jobs=-1,
        )

        # logging best results
        logging.info(f"Best results: {study.best_trial}")
