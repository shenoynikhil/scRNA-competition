import logging
import gc

import numpy as np
import anndata as ad
import pandas as pd

import h5py
import hdf5plugin
import tables

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from cajal import CajalWrapper


cuda = torch.cuda.is_available()


def correlation_score(y_true, y_pred):
    """Scores the predictions according to the competition rules.

    It is assumed that the predictions are not constant.

    Returns the average of each sample's Pearson correlation coefficient

    Source: https://www.kaggle.com/code/xiafire/lb-t15-msci-multiome-catboostregressor#Predicting
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes are different.")
    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)


def load_data_as_anndata(filepaths, metadata_path):
    """
    Loads the files in <filepaths> as AnnData objects

    Source: https://github.com/openproblems-bio/neurips_2022_saturn_notebooks/blob/main/notebooks/loading_and_visualizing_all_data.ipynb
    """
    metadata_df = pd.read_csv(metadata_path)
    metadata_df = metadata_df.set_index("cell_id")

    adatas = {}
    chunk_size = 10000
    for name, info in filepaths.items():
        filename, filepath = info[0], info[1]
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


def train_one_epoch(model, training_loader, epoch_index):
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


def train_all_epochs(
    model, config, save_models, fold, training_loader, validation_loader
):
    epoch_number = 0
    best_vloss = 1_000_000.0

    for epoch in range(config["model_params"]["epochs"]):
        logging.info("EPOCH {}:".format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(model, training_loader, epoch_number)

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
            model_path = os.path.join(
                config["output_dir"],
                "./models/{}_fold{}_epoch{}".format(model.name, fold, epoch_number),
            )
            torch.save(model.state_dict(), model_path)

        epoch_number += 1


def train(model, config, save_models, fold, x_train, y_train, x_val, y_val):
    """
    Train <model> on training data and validation data
    """
    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)
    x_val = torch.Tensor(x_val)
    y_val = torch.Tensor(y_val)
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    training_loader = DataLoader(train_dataset, shuffle=True, batch_size=1000)
    validation_loader = DataLoader(val_dataset, batch_size=1000)
    train_all_epochs(
        model, config, save_models, fold, training_loader, validation_loader
    )


def setup_model(config):
    if config["model"] == "cajal":
        params = config["model_params"]
        logging.info(f"Setting up RBF based Kernel Regressor: {params}")
        return CajalWrapper(config)
    else:
        raise NotImplementedError
