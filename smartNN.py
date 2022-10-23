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
from sklearn.decomposition import TruncatedSVD

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from base import ExperimentHelper
from utils import correlation_score
from basicNN import BasicNN, atac_de_analysis, gex_de_analysis


cuda = torch.cuda.is_available()

important_cols = ['ENSG00000114013_CD86', 'ENSG00000120217_CD274', 'ENSG00000196776_CD47', 'ENSG00000117091_CD48', 'ENSG00000101017_CD40', 'ENSG00000102245_CD40LG', 'ENSG00000169442_CD52', 'ENSG00000117528_ABCD3', 'ENSG00000168014_C2CD3', 'ENSG00000167851_CD300A', 'ENSG00000167850_CD300C', 'ENSG00000186407_CD300E', 'ENSG00000178789_CD300LB', 'ENSG00000186074_CD300LF', 'ENSG00000241399_CD302', 'ENSG00000167775_CD320', 'ENSG00000105383_CD33', 'ENSG00000174059_CD34', 'ENSG00000135218_CD36', 'ENSG00000104894_CD37', 'ENSG00000004468_CD38', 'ENSG00000167286_CD3D', 'ENSG00000198851_CD3E', 'ENSG00000117877_CD3EAP', 'ENSG00000074696_HACD3', 'ENSG00000015676_NUDCD3', 'ENSG00000161714_PLCD3', 'ENSG00000132300_PTCD3', 'ENSG00000082014_SMARCD3', 'ENSG00000121594_CD80', 'ENSG00000110651_CD81', 'ENSG00000238184_CD81-AS1', 'ENSG00000085117_CD82', 'ENSG00000112149_CD83', 'ENSG00000066294_CD84', 'ENSG00000114013_CD86', 'ENSG00000172116_CD8B', 'ENSG00000254126_CD8B2', 'ENSG00000177455_CD19', 'ENSG00000105383_CD33', 'ENSG00000173762_CD7', 'ENSG00000125726_CD70', 'ENSG00000137101_CD72', 'ENSG00000019582_CD74', 'ENSG00000105369_CD79A', 'ENSG00000007312_CD79B', 'ENSG00000090470_PDCD7', 'ENSG00000119688_ABCD4', 'ENSG00000010610_CD4', 'ENSG00000101017_CD40', 'ENSG00000102245_CD40LG', 'ENSG00000026508_CD44', 'ENSG00000117335_CD46', 'ENSG00000196776_CD47', 'ENSG00000117091_CD48', 'ENSG00000188921_HACD4', 'ENSG00000150593_PDCD4', 'ENSG00000203497_PDCD4-AS1', 'ENSG00000115556_PLCD4', 'ENSG00000026508_CD44', 'ENSG00000170458_CD14', 'ENSG00000117281_CD160', 'ENSG00000177575_CD163', 'ENSG00000135535_CD164', 'ENSG00000091972_CD200', 'ENSG00000163606_CD200R1', 'ENSG00000206531_CD200R1L', 'ENSG00000182685_BRICD5', 'ENSG00000111731_C2CD5', 'ENSG00000169442_CD52', 'ENSG00000143119_CD53', 'ENSG00000196352_CD55', 'ENSG00000116815_CD58', 'ENSG00000085063_CD59', 'ENSG00000105185_PDCD5', 'ENSG00000255909_PDCD5P1', 'ENSG00000145284_SCD5', 'ENSG00000167775_CD320', 'ENSG00000110848_CD69', 'ENSG00000139187_KLRG1', 'ENSG00000139193_CD27', 'ENSG00000215039_CD27-AS1', 'ENSG00000120217_CD274', 'ENSG00000103855_CD276', 'ENSG00000204287_HLA-DRA', 'ENSG00000196126_HLA-DRB1', 'ENSG00000198502_HLA-DRB5', 'ENSG00000229391_HLA-DRB6', 'ENSG00000116815_CD58', 'ENSG00000168329_CX3CR1', 'ENSG00000272398_CD24', 'ENSG00000122223_CD244', 'ENSG00000198821_CD247', 'ENSG00000122223_CD244', 'ENSG00000177575_CD163', 'ENSG00000112149_CD83', 'ENSG00000185963_BICD2', 'ENSG00000157617_C2CD2', 'ENSG00000172375_C2CD2L', 'ENSG00000116824_CD2', 'ENSG00000091972_CD200', 'ENSG00000163606_CD200R1', 'ENSG00000206531_CD200R1L', 'ENSG00000012124_CD22', 'ENSG00000150637_CD226', 'ENSG00000272398_CD24', 'ENSG00000122223_CD244', 'ENSG00000198821_CD247', 'ENSG00000139193_CD27', 'ENSG00000215039_CD27-AS1', 'ENSG00000120217_CD274', 'ENSG00000103855_CD276', 'ENSG00000198087_CD2AP', 'ENSG00000169217_CD2BP2', 'ENSG00000144554_FANCD2', 'ENSG00000206527_HACD2', 'ENSG00000170584_NUDCD2', 'ENSG00000071994_PDCD2', 'ENSG00000126249_PDCD2L', 'ENSG00000049883_PTCD2', 'ENSG00000186193_SAPCD2', 'ENSG00000108604_SMARCD2', 'ENSG00000185561_TLCD2', 'ENSG00000075035_WSCD2', 'ENSG00000150637_CD226', 'ENSG00000110651_CD81', 'ENSG00000238184_CD81-AS1', 'ENSG00000134061_CD180', 'ENSG00000004468_CD38', 'ENSG00000012124_CD22', 'ENSG00000150637_CD226', 'ENSG00000135404_CD63', 'ENSG00000135218_CD36', 'ENSG00000137101_CD72', 'ENSG00000125810_CD93', 'ENSG00000010278_CD9', 'ENSG00000125810_CD93', 'ENSG00000153283_CD96', 'ENSG00000002586_CD99', 'ENSG00000102181_CD99L2', 'ENSG00000223773_CD99P1', 'ENSG00000204592_HLA-E', 'ENSG00000085117_CD82', 'ENSG00000134256_CD101']
important_cols = set(important_cols)


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
        day_data = x[x.obs['day'] == day]
        if self.config['technology'] == 'multiome':
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
        day_data = x[x.obs['day'] == day]
        pca_x = TruncatedSVD(
            n_components=output_dim,
            random_state=self.seed
        )
        x_transformed = pca_x.fit_transform(day_data.X)
        return np.average(x_transformed, axis=0)
    
    def _get_context_vector(self, x, combined_train_test, output_dim):
        """
        Stack the context vector for each data point.
        Use the combined train and test data to produce the vector because it can
        provide more context?
        """
        days = x.obs['day']
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

        if self.config['technology'] == "multiome":
            genes = atac_de_analysis(x_train.copy(), self.config['preprocess_params']['top_genes'])
            genes.to_csv(join(self.config["output_dir"], "DEGs.csv"))
            selected_genes = set(genes.names)
        else:
            genes1 = gex_de_analysis(x_train.copy(), self.config['preprocess_params']['top_genes'])
            genes1.to_csv(join(self.config["output_dir"], "DEGs.csv"))
            selected_genes = set(genes1.names).union(y.var_names).union(important_cols)

        logging.info("Getting context vectors")
        combined_x_train_test = ad.concat([x_train, x_test])
        x_train_context = self._get_context_vector(x_train, combined_x_train_test,
            self.config['preprocess_params']['context_output_dim'])
        x_test_context = self._get_context_vector(x_test, combined_x_train_test,
            self.config['preprocess_params']['context_output_dim'])
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
        logging.info(f"{means.shape}, {sds.shape}, {x_train_final.shape}, {x_test_final.shape}")

        logging.info("Dumping means and sds")
        with open(join(self.config["output_dir"], "./transformation.pkl"), "wb") as out:
            pickle.dump(info, out, -1)

        x_train_final = (x_train_final - means) / sds
        x_train_final = x_train_final.T

        x_test_final = x_test_final.T
        x_test_final = (x_test_final - info["means"]) / info["sds"]
        x_test_final = x_test_final.T

        for i in range(self.config['preprocess_params']['context_stack']):
            x_train_final = np.hstack([x_train_final, x_train_context])
            x_test_final = np.hstack([x_test_final, x_test_context])
            gc.collect()

        gc.collect()

        self.input_shape = x_train_final.shape[1]
        self.output_shape = y_final.shape[1]

        return x_train_final, x_test_final, y_final
