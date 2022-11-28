import gc
import logging
import pickle
from os.path import join

import numpy as np

from shallowKFold import ShallowModelKFold
from utils import correlation_score


class SmartKFold(ShallowModelKFold):
    """Takes into input cross val splits provided by user"""

    def read_data(self):
        x_train_transformed, y = super().read_data()
        # also read indices data
        self.x_indices = np.load(self.config["paths"]["x_indices"], allow_pickle=True)[
            "index"
        ].tolist()
        self.cv_path = self.config["paths"].get("cv_file", None)
        if self.cv_path is None:
            raise FileNotFoundError(f"cv_file not found at {self.cv_path}")

        logging.info(f"Setting up cross validation from file: {self.cv_path}")
        with open(self.cv_path, "rb") as f:
            self.cv_splits = pickle.load(f)

        logging.info("Loading test indices")
        eval_indices_cell_ids = np.load(
            self.config["paths"]["eval_indices_path"], allow_pickle=True
        ).tolist()
        self.test_indices = [
            i for i, x in enumerate(self.x_indices) if x in eval_indices_cell_ids
        ]

        return x_train_transformed, y

    def fit_model(self, x, y, y_orig, model, pca_y):
        # construct test set based on test_indices
        x_test, y_test_orig = (
            x[self.test_indices, :],
            y_orig[self.test_indices, :],
        )

        # list for scores and test_scores
        scores, test_scores = [], []
        for i, (cv_split, split_dict) in enumerate(self.cv_splits.items()):
            # train ids and val ids to be used --> convert to set
            train_ids_set = set(split_dict["train"])
            val_ids_set = set(split_dict["val"])

            # get indices
            tr_indices, val_indices = (
                [i for i, x in enumerate(self.x_indices) if x in train_ids_set],
                [i for i, x in enumerate(self.x_indices) if x in val_ids_set],
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
            model = self._fit_model(
                model, x_train, y_train, x_val, y[val_indices, :], pca_y
            )
            del x_train, y_train

            # save models if save_models=True
            if self.save_models:
                pkl_filename = join(self.config["output_dir"], f"model_{i}th_fold.pkl")
                with open(pkl_filename, "wb") as file:
                    pickle.dump(model, file)

            score = correlation_score(y_val, model.predict(x_val) @ pca_y.components_)
            logging.info(f"Score for this val set: {score}")
            scores.append(score)

            # calculate score for test set
            test_score = correlation_score(
                y_test_orig, model.predict(x_test) @ pca_y.components_
            )
            logging.info(f"Score on test set for this split: {test_score}")
            test_scores.append(test_score)

            del x_val, y_val
            gc.collect()
            break

        # Again garbage collection to reduce unnecessary memory usage
        gc.collect()

        # log mean CV scores
        logging.info(f"Mean Test Score: {np.mean(test_scores)}")
        logging.info(f"Mean CV Score: {np.mean(scores)}")

        # log all scores
        logging.info(f"Scores on CV: {scores}")

        # return CV score
        return np.mean(scores)
