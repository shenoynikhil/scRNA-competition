"""To run this script, sample command
python run.py --path=<insert-yaml-path>
"""
import argparse
import gc
import logging
import pickle
from datetime import datetime
from os import makedirs
from os.path import dirname, join
from pathlib import Path

import numpy as np
import scipy
import yaml
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from utils import correlation_score, setup_model


def main(config):
    # part of an HPO trial
    trial = config.get("trial", None)

    # Load Data
    logging.info("Loading data")
    x_train_transformed = pickle.load(open(config["paths"]["x"], "rb"))
    x_test_transformed = pickle.load(open(config["paths"]["x_test"], "rb"))

    # load y as it is, since we need the original values to get metrics
    y = scipy.sparse.load_npz(config["paths"]["y"])

    # perform preprocessing
    logging.info("Performing Preprocessing on y")
    if config["preprocessing"] == "TruncatedSVD":
        # transform y
        pca_y = TruncatedSVD(
            n_components=config["preprocessing_params"]["output_dim"],
            random_state=config["seed"],
        )
        y_transformed = pca_y.fit_transform(y)
    else:
        raise NotImplementedError
    gc.collect()

    # perform KFold cross validation
    logging.info("Setting up cross validation")
    np.random.seed(config["seed"])
    all_row_indices = np.arange(x_train_transformed.shape[0])
    np.random.shuffle(all_row_indices)

    kf = KFold(
        n_splits=config.get('folds', 5), 
        shuffle=True, 
        random_state=config["seed"]
    )

    # Setup model
    logging.info("Setting up the model")
    model = setup_model(config)

    # Save the model (do not save when config['save_models'] == False)
    save_models = config.get("save_models", True)

    # for cross val scores
    scores = []

    # for x_test predictions
    predictions = []
    for i, (tr_indices, val_indices) in enumerate(kf.split(all_row_indices)):
        # preparing ith fold, for y_val we will use (not)transformed vector to calculate scores
        logging.info(f"{i}th fold")
        x_train, y_train = (
            x_train_transformed[tr_indices, :],
            y_transformed[tr_indices, :],
        )
        x_val, y_val = x_train_transformed[val_indices, :], y[val_indices, :].toarray()

        # fit and then delete training splits
        logging.info(f"Fitting the model for {i}th Fold")
        model.fit(x_train, y_train)
        del x_train, y_train

        # save models if save_models=True
        if save_models:
            if trial is not None:
                pkl_filename = join(
                    config["output_dir"], f"trial_{trial}", f"model_{i}th_fold.pkl"
                )
            else:
                pkl_filename = join(config["output_dir"], f"model_{i}th_fold.pkl")
            with open(pkl_filename, "wb") as file:
                pickle.dump(model, file)

        logging.info("Predicting and calculating metrics")
        scores.append(
            correlation_score(y_val, model.predict(x_val) @ pca_y.components_)
        )

        del x_val, y_val
        gc.collect()

        # Perform test predictions with cross val model
        predictions.append(model.predict(x_test_transformed) @ pca_y.components_)
        if i == 1:
            break

    # Again garbage collection to reduce unnecessary memory usage
    gc.collect()

    logging.info(f"Scores on 5-fold CV: {scores}")

    # post processing final predictions before pickle saving them
    if config.get("prediction_agg", "") == "mean":
        # mean over cross val predictions
        prediction = np.mean(predictions, axis=0)
    else:
        raise NotImplementedError

    # save numpy array
    if config.get("save_test_predictions", True):
        if trial is not None:
            pkl_filename = join(
                config["output_dir"], f"trial_{trial}", f"test_pred.pkl"
            )
        else:
            pkl_filename = join(config["output_dir"], f"test_pred.pkl")
        logging.info(f"Saving Predictions to {pkl_filename}")
        makedirs(dirname(pkl_filename), exist_ok=True)
        with open(pkl_filename, "wb") as file:
            pickle.dump(prediction, file)

    # return mean of cross val scores
    return np.mean(scores)


if __name__ == "__main__":
    # take experiment config input
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument(
        "--path", type=str, required=True, help="Path of the experiment config"
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.path).read_text())

    # Setup output directory
    config["output_dir"] = join(
        config["output_dir"], datetime.now().strftime("%d_%m_%Y-%H_%M")
    )
    makedirs(config["output_dir"])
    logging.basicConfig(
        filename=join(config["output_dir"], config.get("log_dir", "output.log")),
        filemode="a",
        level=logging.INFO,
    )

    # log the config
    logging.info(f'Configuration: {config}')

    # run main with config inputted
    main(config)
