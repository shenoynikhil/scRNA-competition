"""To run this script, sample command
python run.py --path=<insert-yaml-path>
"""
import argparse
import gc
import logging
import pickle
from datetime import datetime
from os import makedirs
from os.path import join
from pathlib import Path

import numpy as np
import scipy
import yaml
from sklearn.model_selection import KFold
from utils import correlation_score, preprocessing, setup_model


def main(config):
    # Setup output directory
    output_dir = join(config["output_dir"], datetime.now().strftime("%d_%m_%Y-%H_%M"))
    makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        filename=join(output_dir, "output.log"), filemode="a", level=logging.INFO
    )

    # Load Data
    logging.info("Loading data")
    x = scipy.sparse.load_npz(config["paths"]["x"])
    y = scipy.sparse.load_npz(config["paths"]["y"])
    x_test = scipy.sparse.load_npz(config["paths"]["x_test"])

    # perform preprocessing
    logging.info("Performing Preprocessing")
    (
        _,
        pca_y,
        x_train_transformed,
        y_transformed,
        y,
        x_test_transformed,
    ) = preprocessing(config, x, y, x_test)
    gc.collect()

    # perform KFold cross validation
    logging.info("Setting up cross validation")
    np.random.seed(config["seed"])
    all_row_indices = np.arange(x.shape[0])
    np.random.shuffle(all_row_indices)

    kf = KFold(n_splits=5, shuffle=True, random_state=config["seed"])

    # Setup model
    logging.info("Setting up the model")
    model = setup_model(config)

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

        # perform garbage collection
        gc.collect()

        # fit and then delete training splits
        logging.info(f"Fitting the model for {i}th iteration")
        if config.get('debug', False):
            x_train, y_train = x_train[:10], y_train[:10]
        model.fit(x_train, y_train)
        del x_train, y_train

        # Save the model
        pkl_filename = join(output_dir, f"model_{i}th_fold.pkl")
        with open(pkl_filename, "wb") as file:
            pickle.dump(model, file)

        print("Predicting and calculating metrics")
        if config.get('debug', False):
            x_val, y_val = x_val[:10], y_val[:10]
        scores.append(
            correlation_score(y_val, model.predict(x_val) @ pca_y.components_)
        )

        del x_val, y_val
        gc.collect()

        # Perform test predictions with cross val model
        if config.get('debug', False):
            x_test_transformed = x_test_transformed[:10]        
        predictions.append(
            model.predict(x_test_transformed) @ pca_y.components_
        )

    # Again garbage collection to reduce unnecessary memory usage
    gc.collect()

    logging.info(f"Scores on 5-fold CV: {scores}")
    
    # post processing final predictions before pickle saving them
    if config.get('prediction_agg', "") == 'mean':
        # mean over cross val predictions
        prediction = np.mean(predictions, axis=0)
    else:
        raise NotImplementedError
        
    # save numpy array
    pkl_filename = join(output_dir, f"test_pred.pkl")
    with open(pkl_filename, "wb") as file:
        pickle.dump(prediction, file)        

if __name__ == "__main__":
    # take experiment config input
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument(
        "--path", type=str, required=True, help="Path of the experiment config"
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.path).read_text())

    # run main with config inputted
    main(config)
