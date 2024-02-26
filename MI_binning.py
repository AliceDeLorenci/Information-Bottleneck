# python3 MI_binning.py setup-<setup_idx>/activations-<timestamp>/ <bin_size>

import sys
import os
import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from dataset import buildDatasets, buildDataLoader, loadMNISTData, loadSyntheticData
from mi import compute_mi, plot_info_plan
from nn import Network, train, test, save_activations
from setups import setup_lookup
from utils import save_setup, load_setup

if __name__ == "__main__":

    path = sys.argv[1]
    assert os.path.exists(path), "Folder does not exist: {}".format(path)

    if len(sys.argv) > 2:
        bin_size = float( sys.argv[2] )
    else:
        bin_size = None

    # Extract the setup index from the path and load setup
    setup_idx = int( path.split("/")[0].split("-")[-1] )
    setup = setup_lookup(setup_idx)

    # Load dataset
    ratio = setup["train_ratio"]  # ratio of the training set to the test set
    if setup["dataset"] == "mnist":
        dataset = buildDatasets( *loadMNISTData(root="data"), ratio=ratio, name="mnist" )
    elif setup["dataset"] == "synthetic":
        dataset = buildDatasets( *loadSyntheticData(file="data/synthetic/var_u.mat"), ratio=ratio, name="synthetic" )

    mi_xt_epochs, mi_ty_epochs, epochs = compute_mi(dataset["full"], 
                                                    path, 
                                                    hidden_activation=setup["hidden_activation"], 
                                                    output_activation=setup["output_activation"],
                                                    interval=1, 
                                                    bin_size=bin_size)
    np.savez_compressed( path+"mi-{}".format(bin_size), mi_xt_epochs=mi_xt_epochs, mi_ty_epochs=mi_ty_epochs, epochs=epochs)

    plot_info_plan(mi_xt_epochs, mi_ty_epochs, epochs)
    plt.savefig(path+"info-plan-{}.png".format(bin_size), dpi=300, bbox_inches="tight")
    plt.show()