import gc
import logging
import pickle
from os import makedirs
from os.path import dirname, join

import numpy as np
import optuna
import scipy
from pytorch_tabnet.metrics import Metric
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold

from base import ExperimentHelper
from utils import correlation_score, get_hypopt_space


class ShallowModelKFold(ExperimentHelper):
    def read_data(self):
        # Load Data
        logging.info("Loading data")
        x_train_transformed = pickle.load(open(self.config["paths"]["x"], "rb"))

        # load y as it is, since we need the original values to get metrics
        y = scipy.sparse.load_npz(self.config["paths"]["y"])
        return x_train_transformed, y

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

    def fit_model(self, x, y, y_orig, model, pca_y):
        # perform KFold cross validation
        logging.info("Setting up cross validation")
        np.random.seed(self.seed)
        all_row_indices = np.arange(x.shape[0])
        np.random.shuffle(all_row_indices)

        kf = KFold(
            n_splits=self.config.get("folds", 5), shuffle=True, random_state=self.seed
        )

        # for x_test predictions
        scores = []
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
            model = self._fit_model(model, x_train, y_train, x_val, y[val_indices, :], pca_y)
            del x_train, y_train

            # save models if save_models=True
            if self.save_models:
                pkl_filename = join(self.config["output_dir"], f"model_{i}th_fold.pkl")
                with open(pkl_filename, "wb") as file:
                    pickle.dump(model, file)

            logging.info("Predicting and calculating metrics")
            scores.append(
                correlation_score(y_val, model.predict(x_val) @ pca_y.components_)
            )

            del x_val, y_val
            gc.collect()

        # Again garbage collection to reduce unnecessary memory usage
        gc.collect()

        # log all scores
        logging.info(f"Scores on CV: {scores}")

        # return CV score
        return np.mean(scores)

    def run_experiment(self):
        # get data
        x_train_transformed, y = self.read_data()

        # perform processing
        y_transformed, y, pca_y = self.perform_preprocessing(y)
        gc.collect()

        # get model
        kwargs = {'pca': pca_y}
        model = self.setup_model(**kwargs)

        # run experiment
        score = self.fit_model(
            x_train_transformed,
            y_transformed,
            y.toarray(),
            model,
            pca_y,
        )

        return score

    def _fit_model(self, model, x_train, y_train, x_val=None, y_val=None, pca_y=None):
        """Fit the model correctly"""
        if self.config["model"] == "tabnet":
            class PCC(Metric):
                def __init__(self):
                    self._name = "pcc"
                    self._maximize = True

                def __call__(self, y_true, y_score):
                    y_true, y_score = y_true @ pca_y.components_, y_score @ pca_y.components_
                    corrsum = 0
                    for i in range(len(y_true)):
                        corrsum += np.corrcoef(y_true[i], y_score[i])[1, 0]        
                    return corrsum / y_true.shape[0]
            
            # perform training
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                eval_metric=[PCC],
                max_epochs=300,
                patience=20,
            )
        elif self.config["model"] == "catboost":
            model.fit(
                x_train, y_train, 
                eval_set=[(x_val, y_val)], 
                verbose=True, 
                early_stopping_rounds=10
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
        x_train_transformed, y = self.read_data()

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
            # get hyperopt parameters and set it to model_params, will be used to set model
            self.config["model_params"] = get_hypopt_space(self.config["model"], trial, self.seed)

            train_len = int(train_subset_frac * x.shape[0])  # let's do random for now
            x_train, y_train, x_val, y_val, y_val_orig = (
                x[:train_len, :],
                y[:train_len, :],
                x[train_len:, :],
                y[train_len:, :],
                y_orig[train_len:, :],
            )
            gc.collect()

            model = self.setup_model()
            model = self._fit_model(model, x_train, y_train, x_val, y_val, pca_y)

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
