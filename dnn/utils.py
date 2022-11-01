def update_config(model_config, trial):
    """Function for updating experiment config accordingly"""
    # create a copy of the config to be returned in the end
    model_type = model_config.get("model_type", "BaseNet")
    if model_type == "BaseNet":
        model_config.update(
            {
                "mse_weight": trial.suggest_categorical(
                    "mse_weight", [0.1 * x for x in range(11)]
                ),
                "pcc_weight": trial.suggest_categorical(
                    "mse_weight", [0.1 * x for x in range(11)]
                ),
                "dropout": trial.suggest_categorical(
                    "dropout", [0.1 * x for x in range(5)]
                ),
            }
        )
    elif model_type == "ContextConditioningNet":
        model_config.update(
            {
                "mse_weight": trial.suggest_categorical(
                    "mse_weight", [0.1 * x for x in range(11)]
                ),
                "pcc_weight": trial.suggest_categorical(
                    "mse_weight", [0.1 * x for x in range(11)]
                ),
                "dropout": trial.suggest_categorical(
                    "dropout", [0.1 * x for x in range(5)]
                ),
                "beta": trial.suggest_float(
                    "beta", low=0.0001, high=1.0, step=0.0001, log=True
                ),
            }
        )

    return model_config
