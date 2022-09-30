import logging
import gc
import numpy as np
import pickle

# sklearn stuff: Model Specific Libraries
from sklearn.model_selection import KFold

import scipy
from utils import PATHS, correlation_score, preprocessing, setup_model

logging.basicConfig(level=logging.INFO)

config = {
    "seed": 42,
    "scale": 10,
    "alpha": 0.1,
    "preprocessing_strategy": "TruncatedSVD",
    "model": "rbf_krr",
    "cite_components_rna": 10,
    "cite_components_proteins": 1,
    "multiome_components_atac": 50,
    "multiome_components_rna": 50,
}


def main():
    # Load Data
    logging.info("Loading data")
    x = scipy.sparse.load_npz(PATHS["train_cite_inputs"])
    y = scipy.sparse.load_npz(PATHS["train_cite_targets"])
    x_test = scipy.sparse.load_npz(PATHS["test_cite_inputs"])

    # perform preprocessing
    logging.info("Performing Preprocessing")
    (
        _,
        pca_y,
        x_train_transformed,
        y_transformed,
        y,
        _,
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

    score = []
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
        model.fit(x_train, y_train)
        del x_train, y_train

        # Save the model
        pkl_filename = f"model_{i}th_fold.pkl"
        with open(pkl_filename, "wb") as file:
            pickle.dump(model, file)

        print("Predicting and calculating metrics")
        score.append(
            correlation_score(y_val.toarray(), model.predict(x_val) @ pca_y.components_)
        )

        del x_val, y_val
        gc.collect()

    # Again garbage collection to reduce unnecessary memory usage
    gc.collect()


if __name__ == "__main__":
    main()
