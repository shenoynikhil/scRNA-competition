"""
Script to run hyperparameter optimization using Optuna
"""
import argparse
import logging
from datetime import datetime
from os import makedirs
from os.path import join
from pathlib import Path

import yaml

from shallowKFold import ShallowModelKFold


def main(config):
    # run main with config inputted
    if config["experiment"] == "ShallowModelKFold":
        experiment = ShallowModelKFold(config)
        experiment.conduct_hpo(n_trials=100, subset_size=10000, train_subset_frac=0.8)
    else:
        raise NotImplementedError


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
    logging.info(f"Configuration: {config}")

    main(config)
