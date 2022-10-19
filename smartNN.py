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


class SmartNN(ExperimentHelper):
    """
    Neural network that incorporates some context vector.
    Regular input can be top-n genes or a PCA.
    Context vector can be avg of previous day inputs or PCA.
    """