import os
import gc

os.environ["NUMBA_CACHE_DIR"] = "/tmp/"  # https://github.com/scverse/scanpy/issues/2113
from os.path import join
import logging
import anndata as ad
import pickle
import numpy as np
import optuna
from sklearn.decomposition import TruncatedSVD

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from utils import correlation_score, get_hypopt_space, important_cols
from basicNN import BasicNN, atac_de_analysis, gex_de_analysis


cuda = torch.cuda.is_available()


class ContextNN(nn.Module):
    """
    Implementation of the smarter NN
    """

    def __init__(self, hp, input_shape, output_shape, min_val, max_val):
        super(ContextNN, self).__init__()
        self.name = "Smart Neural Network"
        modules = [
            nn.Dropout(hp.dropout),
            nn.Linear(input_shape, hp.layer_shapes[0]),
            nn.ReLU(),
        ]
        for i in range(hp.n_layers - 1):
            modules.append(nn.Linear(hp.layer_shapes[i], hp.layer_shapes[i + 1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hp.layer_shapes[-1], output_shape))
        self.stack = nn.Sequential(*modules)
        self.lambd = lambda x: torch.clamp(x, min_val, max_val)

    def forward(self, x):
        x = self.stack(x)
        return self.lambd(x)


class SmartNN(BasicNN):
    """
    Neural network that incorporates some context vector.
    Regular input can be top-n genes or a PCA.
    Context vector can be avg of previous day inputs or PCA.
    """

    def _get_day_average(self, x, day, output_dim):
        """
        Returns the average for the day
        """
        day_data = x[x.obs["day"] == day]
        if self.config["technology"] == "multiome":
            genes = atac_de_analysis(day_data.copy(), output_dim)
            day_data = day_data[:, genes]
        else:
            genes = gex_de_analysis(day_data.copy(), output_dim)
            day_data = day_data[:, genes]
        return np.average(day_data.X.toarray(), axis=0)

    def _get_day_pca(self, x, day, output_dim):
        """
        Returns a PCA of output_dim for day
        """
        day_data = x[x.obs["day"] == day]
        pca_x = TruncatedSVD(n_components=output_dim, random_state=self.seed)
        x_transformed = pca_x.fit_transform(day_data.X)
        return np.average(x_transformed, axis=0)

    def _get_context_vector(self, x, combined_train_test, output_dim):
        """
        Stack the context vector for each data point.
        Use the combined train and test data to produce the vector because it can
        provide more context?
        """
        days = x.obs["day"]
        unique_days = np.unique(days)
        context_vector = np.zeros((x.X.shape[0], output_dim))
        prev_day_dic = {2: 2, 3: 2, 4: 3, 7: 4, 10: 7}
        for day in unique_days:
            logging.info(f"Day {day} context vec")
            prev_day = prev_day_dic[day]
            pca_day = self._get_day_pca(combined_train_test, prev_day, output_dim)
            context_vector[np.where(days == day)[0]] = pca_day
        return context_vector

    def perform_preprocessing(self, x_train, x_test, y):
        logging.info("Preprocessing data")

        if self.config["technology"] == "multiome":
            genes = atac_de_analysis(
                x_train.copy(), self.config["preprocess_params"]["top_genes"]
            )
            genes.to_csv(join(self.config["output_dir"], "DEGs.csv"))
            selected_genes = set(genes.names)
        else:
            genes1 = gex_de_analysis(
                x_train.copy(), self.config["preprocess_params"]["top_genes"]
            )
            genes1.to_csv(join(self.config["output_dir"], "DEGs.csv"))
            selected_genes = set(genes1.names).union(y.var_names).union(important_cols)

        logging.info("Getting context vectors")
        combined_x_train_test = ad.concat([x_train, x_test])
        x_train_context = self._get_context_vector(
            x_train,
            combined_x_train_test,
            self.config["preprocess_params"]["context_output_dim"],
        )
        x_test_context = self._get_context_vector(
            x_test,
            combined_x_train_test,
            self.config["preprocess_params"]["context_output_dim"],
        )
        gc.collect()

        self.min_val = np.min(y.X)
        self.max_val = np.max(y.X)

        subset = selected_genes.intersection(x_train.var_names)
        x_train = x_train[:, list(subset)]
        x_test = x_test[:, list(subset)]

        logging.info("Dumping import genes")
        with open(join(self.config["output_dir"], "genes.pkl"), "wb") as out:
            pickle.dump(x_train.var_names, out, -1)

        logging.info("Calculating summary stats")

        train_total = np.sum(x_train.X.toarray(), axis=1)
        test_total = np.sum(x_test.X.toarray(), axis=1)

        gc.collect()

        train_batches = set(x_train.obs.donor)
        x_train.obs["batch_median"] = 0
        x_train.obs["batch_sd"] = 0
        for batch in train_batches:
            x_train.obs["batch_median"][x_train.obs.donor == batch] = np.median(
                train_total[x_train.obs.donor == batch]
            )
            x_train.obs["batch_sd"][x_train.obs.donor == batch] = np.std(
                train_total[x_train.obs.donor == batch]
            )

        test_batches = set(x_test.obs.donor)
        x_test.obs["batch_median"] = 0
        x_test.obs["batch_sd"] = 0

        for batch in test_batches:
            x_test.obs["batch_median"][x_test.obs.donor == batch] = np.median(
                test_total[x_test.obs.donor == batch]
            )
            x_test.obs["batch_sd"][x_test.obs.donor == batch] = np.std(
                test_total[x_test.obs.donor == batch]
            )

        x_train_final = x_train.X.toarray()
        train_batch_median = x_train.obs["batch_median"]
        train_batch_sd = x_train.obs["batch_sd"]
        train_batch_days = x_train.obs["day"]

        logging.info("Stacking summary stats")
        stack = self.config["preprocess_params"]["stack"]
        for i in range(stack):
            x_train_final = np.column_stack((x_train_final, train_total))
            gc.collect()
        for i in range(stack):
            x_train_final = np.column_stack((x_train_final, train_batch_median))
            gc.collect()
        for i in range(stack):
            x_train_final = np.column_stack((x_train_final, train_batch_sd))
            gc.collect()

        x_test_final = x_test.X.toarray()
        test_batch_median = x_test.obs["batch_median"]
        test_batch_sd = x_test.obs["batch_sd"]
        test_batch_days = x_test.obs["day"]

        for i in range(stack):
            x_test_final = np.column_stack((x_test_final, test_total))
            gc.collect()
        for i in range(stack):
            x_test_final = np.column_stack((x_test_final, test_batch_median))
            gc.collect()
        for i in range(stack):
            x_test_final = np.column_stack((x_test_final, test_batch_sd))
            gc.collect()

        y_final = y.X.toarray()
        del y

        x_train_final = x_train_final.T
        means = np.mean(x_train_final, axis=1)
        sds = np.std(x_train_final, axis=1)
        means = means.reshape(len(means), 1)
        sds = sds.reshape(len(sds), 1)
        info = {"means": means, "sds": sds}
        logging.info(
            f"{means.shape}, {sds.shape}, {x_train_final.shape}, {x_test_final.shape}"
        )

        logging.info("Dumping means and sds")
        with open(join(self.config["output_dir"], "./transformation.pkl"), "wb") as out:
            pickle.dump(info, out, -1)

        x_train_final = (x_train_final - means) / sds
        x_train_final = x_train_final.T

        x_test_final = x_test_final.T
        x_test_final = (x_test_final - info["means"]) / info["sds"]
        x_test_final = x_test_final.T

        for i in range(self.config["preprocess_params"]["context_stack"]):
            x_train_final = np.hstack([x_train_final, x_train_context])
            x_test_final = np.hstack([x_test_final, x_test_context])
            gc.collect()

        gc.collect()

        self.input_shape = x_train_final.shape[1]
        self.output_shape = y_final.shape[1]

        return x_train_final, x_test_final, y_final

    def _opt_fit_model(self, x, x_val, y, y_val, model):
        logging.info("Generating train and validation datasets")
        np.random.seed(self.config["seed"])
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        x_val = torch.Tensor(x_val)
        y_val = torch.Tensor(y_val)
        train_dataset = TensorDataset(x, y)
        val_dataset = TensorDataset(x_val, y_val)
        training_loader = DataLoader(train_dataset, batch_size=1000)
        validation_loader = DataLoader(val_dataset, batch_size=1000)
        self._train_all_epochs(model, training_loader, validation_loader)

    def conduct_hpo(
        self,
        subset_size: int = 5000,
        n_trials: int = 10,
        train_subset_frac: float = 0.8,
    ):
        """
        Conduct hyperparameter optimization?
        """
        x_train, x_test, y_train = self.read_data()
        x_train_final, _, y_final = self.perform_preprocessing(x_train, x_test, y_train)
        gc.collect()

        # Random shuffle
        logging.info("Reducing size")
        p = np.random.permutation(len(x_train_final))
        x_train_final, y_final = x_train_final[p], y_final[p]
        x_train_final, y_final = (
            x_train_final[:subset_size, :],
            y_final[:subset_size, :],
        )
        gc.collect()

        def objective(trial, x, y):
            logging.info(str(trial))
            self.config["model_params"] = get_hypopt_space(
                self.config["model"], trial, self.seed
            )

            train_len = int(train_subset_frac * x.shape[0])
            x_train, y_train, x_val, y_val = (
                x[:train_len, :],
                y[:train_len, :],
                x[train_len:, :],
                y[train_len:, :],
            )
            gc.collect()

            model = self.setup_model()
            self._opt_fit_model(x, x_val, y, y_val, model)

            model.eval()

            x_val = torch.Tensor(x_val)
            if cuda:
                x_val = x_val.to("cuda")
            pred_y_val = model(x_val)
            pred_y_val = pred_y_val.cpu().detach().numpy()
            score = correlation_score(y_val, pred_y_val)
            logging.info(f"Score for trial: {trial} is, {score}")
            return score

        logging.info("Running hyperopt")
        study = optuna.create_study(study_name="hpo-run", direction="maximize")
        study.optimize(
            lambda trial: objective(trial, x_train_final, y_final),
            n_trials=n_trials,
            n_jobs=1,
            timeout=42000
        )

        # logging best results
        logging.info(f"Best results: {study.best_trial}")
