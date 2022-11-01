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
from smartNN import SmartNN
from dnn import DNNSetup


def main(config):
    # Setup output directory
    config["output_dir"] = join(
        config["output_dir"], 'hpo', datetime.now().strftime("%d_%m_%Y-%H_%M")
    )
    makedirs(config["output_dir"])
    logging.basicConfig(
        filename=join(config["output_dir"], config.get("log_dir", "output.log")),
        filemode="a",
        level=logging.INFO,
    )

    # run main with config inputted
    n_trials = config.get('n_trials', 100)
    if config["experiment"] == "ShallowModelKFold":
        experiment = ShallowModelKFold(config)
        experiment.conduct_hpo(n_trials=n_trials, subset_size=10000, train_subset_frac=0.8)
    elif config["experiment"] == "SmartNeuralNetwork":
        experiment = SmartNN(config)
        experiment.conduct_hpo(n_trials=n_trials, subset_size=10000, train_subset_frac=0.8)
    elif config["experiment"] == "DNN":
        experiment = DNNSetup(config)
        experiment.conduct_hpo(n_trials=n_trials)
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
    main(config)
