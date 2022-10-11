import os
import gc

os.environ["NUMBA_CACHE_DIR"] = "/tmp/"  # https://github.com/scverse/scanpy/issues/2113
from os.path import basename, join
from os import makedirs

import logging
import anndata as ad
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import scipy

import h5py
import hdf5plugin
import tables

from sklearn.preprocessing import binarize

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from base import ExperimentHelper
from utils import correlation_score


cuda = torch.cuda.is_available()


def atac_de_analysis(adata):
    """get top DA peaks per cell type"""
    adata.X = binarize(adata.X)
    sc.tl.rank_genes_groups(adata, "cell_type", method="t-test")
    cell_types = adata.obs.cell_type.value_counts().index
    column_names = [
        "names",
        "scores",
        "logfoldchanges",
        "pvals",
        "pvals_adj",
        "cell_type",
    ]
    df = pd.DataFrame(columns=column_names)
    for cell_type in cell_types:
        dedf = sc.get.rank_genes_groups_df(adata, group=cell_type)
        dedf["cell_type"] = cell_type
        dedf = dedf.sort_values("scores", ascending=False).iloc[:100]
        df = df.append(dedf, ignore_index=True)
    return df


def gex_de_analysis(adata_GEX):
    """get top DE genes per cell type (multiome)"""
    sc.pp.filter_cells(adata_GEX, min_genes=200)
    sc.pp.filter_genes(adata_GEX, min_cells=3)
    adata_GEX.var["mt"] = adata_GEX.var_names.str.contains("MT-")
    sc.pp.calculate_qc_metrics(
        adata_GEX, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    adata_GEX = adata_GEX[adata_GEX.obs.n_genes_by_counts < 4000, :]
    sc.pp.normalize_total(adata_GEX, target_sum=1e4)
    sc.pp.log1p(adata_GEX)
    sc.pp.highly_variable_genes(adata_GEX, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pp.scale(adata_GEX, max_value=10)
    sc.tl.rank_genes_groups(adata_GEX, "cell_type", method="wilcoxon")
    cell_types = adata_GEX.obs.cell_type.value_counts().index
    column_names = [
        "names",
        "scores",
        "logfoldchanges",
        "pvals",
        "pvals_adj",
        "cell_type",
    ]
    df = pd.DataFrame(columns=column_names)
    for cell_type in cell_types:
        dedf = sc.get.rank_genes_groups_df(adata_GEX, group=cell_type)
        dedf["cell_type"] = cell_type
        dedf = dedf.sort_values("scores", ascending=False).iloc[:100]
        df = df.append(dedf, ignore_index=True)
    return df


def load_data_as_anndata(filepaths, metadata_path):
    """
    Loads the files in <filepaths> as AnnData objects

    Source: https://github.com/openproblems-bio/neurips_2022_saturn_notebooks/blob/main/notebooks/loading_and_visualizing_all_data.ipynb
    """
    metadata_df = pd.read_csv(metadata_path)
    metadata_df = metadata_df.set_index("cell_id")

    adatas = {}
    chunk_size = 10000
    for name, filepath in filepaths.items():
        filename = basename(filepath)[:-3]
        logging.info(f"Loading {filename}")

        h5_file = h5py.File(filepath)
        h5_data = h5_file[filename]

        features = h5_data["axis0"][:]
        cell_ids = h5_data["axis1"][:]

        features = features.astype(str)
        cell_ids = cell_ids.astype(str)

        technology = metadata_df.loc[cell_ids, "technology"].unique().item()

        if technology == "multiome":
            sparse_chunks = []
            n_cells = h5_data["block0_values"].shape[0]

            for chunk_indices in np.array_split(np.arange(n_cells), 100):
                chunk = h5_data["block0_values"][chunk_indices]
                sparse_chunk = scipy.sparse.csr_matrix(chunk)
                sparse_chunks.append(sparse_chunk)

            X = scipy.sparse.vstack(sparse_chunks)
        elif technology == "citeseq":
            X = h5_data["block0_values"][:]

        adata = ad.AnnData(
            X=X,
            obs=metadata_df.loc[cell_ids],
            var=pd.DataFrame(index=features),
        )

        adatas[name] = adata

    return adatas


class Hyperparameters:
    # class to store hyperparameters
    def __init__(self, dropout, layer_shapes):
        self.dropout = dropout
        self.layer_shapes = layer_shapes
        self.n_layers = len(layer_shapes)


class Cajal(nn.Module):
    def __init__(self, hp, input_shape, output_shape, min_val, max_val):
        super(Cajal, self).__init__()
        self.name = "Cajal"
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


class BasicNN(ExperimentHelper):
    '''
    Basic Neural Network using Team Cajal solution from 2021
    '''
    def read_data(self):
        logging.info("Loading data")

        adatas = load_data_as_anndata(self.config['paths'], self.config['metadata'])

        return adatas['x'], adatas['x_test'], adatas['y']
    
    def perform_preprocessing(self, x_train, x_test, y):
        logging.info("Preprocessing data")

        if self.config['technology'] == "multiome":
            genes = atac_de_analysis(x_train.copy())
            genes.to_csv(join(self.config["output_dir"], "DEGs.csv"))
            selected_genes = set(genes.names)
        else:
            genes1 = gex_de_analysis(x_train.copy())
            genes1.to_csv(join(self.config["output_dir"], "DEGs.csv"))
            selected_genes = set(genes1.names).union(y.var_names)

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
        del x_train
        gc.collect()

        logging.info("Stacking summary stats")
        stack = self.config['preprocess_params']['stack']
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

        del x_test
        gc.collect()

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

        logging.info("Dumping means and sds")
        with open(join(self.config["output_dir"], "./transformation.pkl"), "wb") as out:
            pickle.dump(info, out, -1)

        x_train_final = (x_train_final - means) / sds
        x_train_final = x_train_final.T

        x_test_final = x_test_final.T
        x_test_final = (x_test_final - info["means"]) / info["sds"]
        x_test_final = x_test_final.T

        self.input_shape = x_train_final.shape[1]
        self.output_shape = y_final.shape[1]

        return x_train_final, x_test_final, y_final

    
    def setup_model(self):
        torch.manual_seed(self.config['seed'])
        hp = Hyperparameters(self.config['model_params']['dropout'], self.config['model_params']['layers'])
        model = Cajal(hp, self.input_shape, self.output_shape, self.min_val, self.max_val)
        if cuda:
            model.to('cuda')
        model.train()
        return model

    def _train_one_epoch(self, model, training_loader, epoch_index, loss_fn, optimizer):
        """
        Literally the most basic training epoch
        """
        running_loss = 0.0
        last_loss = 0.0

        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data

            if cuda:
                inputs = inputs.to("cuda")
                labels = labels.to("cuda")

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            last_loss = running_loss
            logging.info("  batch {} loss: {}".format(i + 1, running_loss))
            running_loss = 0.0
            del inputs, labels
            gc.collect()

        return last_loss

    def _train_all_epochs(self, model, training_loader, validation_loader):
        save_models = self.config.get("save_test_predictions", True)
        epoch_number = 0
        best_vloss = 1_000_000.0

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(self.config["model_params"]["epochs"]):
            logging.info("EPOCH {}:".format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            avg_loss = self._train_one_epoch(model, training_loader, epoch_number, loss_fn, optimizer)

            # We don't need gradients on to do reporting
            model.train(False)

            running_vloss = 0.0
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                if cuda:
                    vinputs = vinputs.to("cuda")
                    vlabels = vlabels.to("cuda")
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss
                del vinputs, vlabels
                gc.collect()

            avg_vloss = running_vloss / (i + 1)
            logging.info("LOSS train {} valid {}".format(avg_loss, avg_vloss))

            # Track best performance, and save the model's state
            if save_models and avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = join(
                    self.config["output_dir"],
                    "{}_epoch{}".format("BasicNN", epoch_number),
                )
                torch.save(model.state_dict(), model_path)

            epoch_number += 1

    def fit_model(self, x, y, model):
        logging.info("Generating train and validation datasets")
        np.random.seed(self.config['seed'])
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        dataset = TensorDataset(x, y)
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [90807, 15135], generator=torch.Generator().manual_seed(self.config['seed'])
        )
        training_loader = DataLoader(train_dataset, batch_size=1000)
        validation_loader = DataLoader(val_dataset, batch_size=1000)
        self._train_all_epochs(model, training_loader, validation_loader)

    def predict(self, x_test, model):
        logging.info("Predicting for test data")
        model.eval()
        x_test = torch.Tensor(x_test)
        if cuda:
            x_test = x_test.to('cuda')
        y_test = model(x_test)
        if cuda:
            y = y_test.cpu().detach().numpy()
        else:
            y = y_test.detach().numpy()
        if self.config.get("save_test_predictions", True):
            pkl_filename = join(self.config["output_dir"], f"test_pred.pkl")
            logging.info(f"Saving Predictions to {pkl_filename}")
            makedirs(dirname(pkl_filename), exist_ok=True)
            with open(pkl_filename, "wb") as file:
                pickle.dump(prediction, file)

    def _load_and_predict(self, x_test):
        model = self.setup_model()
        if not cuda:
            model.load_state_dict(torch.load(self.config['load_path']), map_location=torch.device('cpu'))
        else:
            model.load_state_dict(torch.load(self.config['load_path']))
        model.eval()
        x_test = torch.Tensor(x_test)
        if cuda:
            x_test = x_test.to('cuda')
        y_test = model(x_test)
        if cuda:
            y = y_test.cpu().detach().numpy()
        else:
            y = y_test.detach().numpy()
        if self.config.get("save_test_predictions", True):
            pkl_filename = join(self.config["output_dir"], f"test_pred.pkl")
            logging.info(f"Saving Predictions to {pkl_filename}")
            # makedirs(dirname(pkl_filename), exist_ok=True)
            with open(pkl_filename, "wb") as file:
                pickle.dump(y, file)

    
    def run_experiment(self):
        x, x_test, y = self.read_data()
        x_train_final, x_test_final, y_final = self.perform_preprocessing(x, x_test, y)
        logging.info(f"Cuda is available: {cuda}")

        if not self.config['load']:
            logging.info("Setting up and training new model")
            model = self.setup_model()
            self.fit_model(x_train_final, y_final , model)
            self.predict(x_test_final, model)
        else:
            logging.info(f"Loading model from { self.config['load_path']}")
            self._load_and_predict(x_test_final)
        logging.info("Completed")
