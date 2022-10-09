"""To run this script, sample command
python svd.py --path=<insert-yaml-path>
"""
import argparse
import gc
import logging
import pickle
from datetime import datetime
from os import makedirs
from os.path import join
from pathlib import Path

import yaml
from scipy.sparse import load_npz
import scipy
from sklearn.decomposition import TruncatedSVD

SAVE_DIR = "/scratch/st-jiaruid-1/shenoy/svd-comp"


def main(config):
    # Load Data
    logging.info("Loading data")
    x = load_npz(config["paths"]["x"])
    x_test = load_npz(config["paths"]["x_test"])

    # perform preprocessing
    # transform x and x_test
    pca_x = TruncatedSVD(
        n_components=config["preprocessing_params"]["input_dim"],
        random_state=config["seed"],
    )
    x_stacked = scipy.sparse.vstack([x, x_test])
    x_transformed = pca_x.fit_transform(x_stacked)

    x_train_transformed = x_transformed[: x.shape[0], :]
    x_test_transformed = x_transformed[x.shape[0] :, :]    
    del x, x_test
    gc.collect()

    # save the processed arrays
    logging.info("Saving")
    input_dim = config["preprocessing_params"]["input_dim"]
    input_type = config["type"]
    pickle.dump(
        x_train_transformed,
        open(join(SAVE_DIR, f"train_input_{input_type}_svd{input_dim}.pkl"), "wb"),
    )
    pickle.dump(
        x_test_transformed,
        open(join(SAVE_DIR, f"test_input_{input_type}_svd{input_dim}.pkl"), "wb"),
    )


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
    makedirs(config["output_dir"], exist_ok=True)
    logging.basicConfig(
        filename=join(config["output_dir"], config.get("log_dir", "output.log")),
        filemode="a",
        level=logging.INFO,
    )

    # run main with config inputted
    main(config)
