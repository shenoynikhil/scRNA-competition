import gc
import logging
import pickle
from os import makedirs
from os.path import dirname, join

import numpy as np

from shallowKFold import ShallowModelKFold
from utils import correlation_score


class SmartKFold(ShallowModelKFold):
    def read_data(self):
        x_train_transformed, x_test_transformed, y = super().read_data()
        # also read indices data
        self.x_indices = np.load(self.config["paths"]["x_indices"], allow_pickle=True)[
            "index"
        ].tolist()
        self.cv_path = self.config.get("cv_file", None)
        if self.cv_path is None:
            raise FileNotFoundError(f"cv_file not found at {self.cv_path}")

        return x_train_transformed, x_test_transformed, y

    def fit_model(self, x, y, y_orig, x_test, model, pca_y):
        logging.info(f"Setting up cross validation from file: {self.cv_path}")
        with open(self.cv_path, "rb") as f:
            cv_splits = pickle.load(f)

        # Save the model (do not save when config['save_models'] == False)
        save_models = self.config.get("save_models", True)

        # for cross val scores
        scores = []

        # for x_test predictions
        predictions = []
        for i, (cv_split, split_dict) in enumerate(cv_splits.items()):
            # train ids and val ids to be used --> convert to set
            train_ids_set = set(split_dict["train"])
            val_ids_set = set(split_dict["val"])

            # get indices
            tr_indices, val_indices = (
                len([i for i, x in enumerate(self.x_indices) if x in train_ids_set]),
                len([i for i, x in enumerate(self.x_indices) if x in val_ids_set]),
            )

            # preparing ith fold, for y_val we will use (not)transformed vector to calculate scores
            logging.info(f"{cv_split}th fold")
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
