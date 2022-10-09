import os

os.environ["NUMBA_CACHE_DIR"] = "/tmp/"  # https://github.com/scverse/scanpy/issues/2113
from os.path import join

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


class CajalWrapper:
    """
    Wrapper for the neural network from Team Cajal
    """

    def __init__(self, config):
        self.type = config["type"]
        self.config = config

    def preprocess(self, x_train: ad.AnnData, x_test: ad.AnnData, y: ad.AnnData):
        if self.type == "multiome":
            genes = atac_de_analysis(x_train.copy())
            genes.to_csv(join(self.config["output_dir"], "DEGs.csv"))
            selected_genes = set(genes.names)
        else:
            genes1 = gex_de_analysis(x_train.copy())
            genes1.to_csv(join(self.config["output_dir"], "DEGs.csv"))
            selected_genes = set(genes1.names).union(y.var_names)

        self.min_val = np.min(y.X)
        self.max_val = np.max(y.X)

        train_total = np.sum(x_train.X.toarray(), axis=1)
        test_total = np.sum(x_test.X.toarray(), axis=1)

        subset = selected_genes.intersection(x_train.var_names)
        x_train = x_train[:, list(subset)]

        with open(join(self.config["output_dir"], "genes.pkl"), "wb") as out:
            pickle.dump(x_train.var_names, out, -1)

        x_train_final = x_train.X.toarray()
        y_final = y.X.toarray()
        x_test_final = x_test.X.toarray()

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

        for i in range(50):
            x_train_final = np.column_stack((X_train, train_total))
        for i in range(50):
            x_train_final = np.column_stack((X_train, x_train.obs["batch_median"]))
        for i in range(50):
            x_train_final = np.column_stack((X_train, x_train.obs["batch_sd"]))

        for i in range(50):
            x_test_final = np.column_stack((x_test_final, test_total))
        for i in range(50):
            x_test_final = np.column_stack((x_test_final, x_test.obs["batch_median"]))
        for i in range(50):
            x_test_final = np.column_stack((x_test_final, x_test.obs["batch_sd"]))

        x_train_final = x_train_final.T
        means = np.mean(x_train_final, axis=1)
        sds = np.std(x_train_final, axis=1)
        means = means.reshape(len(means), 1)
        sds = sds.reshape(len(sds), 1)
        info = {"means": means, "sds": sds}

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

    def initialize(self) -> Cajal:
        """
        Setup a Cajal model using the hyperparameters in <config>
        """
        params = config["model_params"]
        hp = Hyperparameters(params["dropout"], params["hidden_layers"])
        return Cajal(
            hp, self.input_shape, self.output_shape, self.min_val, self.max_val
        )
