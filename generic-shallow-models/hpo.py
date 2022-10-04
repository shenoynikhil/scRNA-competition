"""
Script to run hyperparameter optimization using Optuna
"""
import argparse
import logging
from datetime import datetime
from os import makedirs
from os.path import join
from pathlib import Path

import optuna
import yaml
from run import main


def objective(trial, config):
    # setup hyperparams
    hparams_dict = config["hpo"]["params"]
    for param_header, param_values in hparams_dict.items():
        for val, val_setting in param_values.items():
            if val_setting["type"] == "float":
                config[param_header][val] = trial.suggest_float(
                    name=f"{param_header}/{val}",
                    low=val_setting["low"],
                    high=val_setting["high"],
                    step=val_setting["step"],
                )
            elif val_setting["type"] == "int":
                config[param_header][val] = trial.suggest_int(
                    name=f"{param_header}/{val}",
                    low=val_setting["low"],
                    high=val_setting["high"],
                    step=val_setting["step"],
                )
            else:
                raise NotImplementedError

    # save trial number in config
    config["trial"] = trial.number

    return main(config)


if __name__ == "__main__":
    # take experiment config input
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument(
        "--path", type=str, required=True, help="Path of the experiment config"
    )
    args = parser.parse_args()

    # read config input
    config = yaml.safe_load(Path(args.path).read_text())

    # read output_dir
    config["output_dir"] = join(
        config["output_dir"], datetime.now().strftime("%d_%m_%Y-%H_%M")
    )
    makedirs(config["output_dir"], exist_ok=True)
    logging.basicConfig(
        filename=join(config["output_dir"], config.get("log_dir", "output.log")),
        filemode="a",
        level=logging.INFO,
    )

    study = optuna.create_study(
        study_name=config["hpo"].get("study_name", "HPO"), direction="maximize"
    )
    study.optimize(
        lambda trial: objective(trial, config), 
        n_trials=config["hpo"]["trials"],
        n_jobs=2
    )
    logging.info(f"Best results: {study.best_trial}")
