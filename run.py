"""
To run this script, sample command
python run.py --path=<insert-yaml-path>
"""
import argparse
import logging
from datetime import datetime
from os import makedirs
from os.path import join
from pathlib import Path

import yaml

from basicNN import BasicNN
from dnn import DNNSetup
from shallowKFold import ShallowModelKFold
from smartKFold import SmartKFold
from smartNN import SmartNN


def main(config):
    """Runs the experiment, if you have a new experiment type, import above
    and add an elif statement corresponding to it.
    """
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

    # log the config
    logging.info(f"Configuration: {config}")

    # run main with config inputted
    if config["experiment"] == "ShallowModelKFold":
        experiment = ShallowModelKFold(config)
        experiment.run_experiment()
    elif config["experiment"] == "SmartKFold":
        experiment = SmartKFold(config)
        experiment.run_experiment()
    elif config["experiment"] == "BasicNeuralNetwork":
        experiment = BasicNN(config)
        experiment.run_experiment()
    elif config["experiment"] == "SmartNeuralNetwork":
        experiment = SmartNN(config)
        experiment.run_experiment()
    elif config["experiment"] == "DNN":
        experiment = DNNSetup(config)
        experiment.run_experiment()
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
