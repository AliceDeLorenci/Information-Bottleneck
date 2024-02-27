# python3 MI.py setup-<setup_idx>/activations-<timestamp>/ <bin_size>

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


BINNING_ESTIMATOR = False   # whether to use binning estimator (True) or KDE (False)
INTERVAL = 1                # interval for computing MI

if __name__ == "__main__":

    path = sys.argv[1]
    assert os.path.exists(path), "Folder does not exist: {}".format(path)

    if len(sys.argv) > 2:
        bin_size = float( sys.argv[2] )
    else:
        bin_size = None

    # Choose torch device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")     
    print("Using device:", device)

    # Extract the setup index from the path and load setup
    setup_idx = int( path.split("/")[0].split("-")[-1] )
    setup = setup_lookup(setup_idx)

    # Load dataset
    ratio = setup["train_ratio"]  # ratio of the training set to the test set
    if setup["dataset"] == "mnist":
        dataset = buildDatasets( *loadMNISTData(root="data"), ratio=ratio, name="mnist" )
    elif setup["dataset"] == "synthetic":
        dataset = buildDatasets( *loadSyntheticData(file="data/synthetic/var_u.mat"), ratio=ratio, name="synthetic" )

    mi_xt_epochs, mi_ty_epochs, epochs = compute_mi(dataset["full"] if setup["dataset"] == "synthetic" else dataset["test"], 
                                                    setup,
                                                    path, 
                                                    interval=INTERVAL, 
                                                    bin_size=bin_size,
                                                    device=device,
                                                    binning_estimator=BINNING_ESTIMATOR)
    
    np.savez_compressed( path+"mi-{}".format(bin_size if BINNING_ESTIMATOR else "kde"), mi_xt_epochs=mi_xt_epochs, mi_ty_epochs=mi_ty_epochs, epochs=epochs)

    plot_info_plan(mi_xt_epochs, mi_ty_epochs, epochs)
    plt.savefig(path+"info-plan-{}.png".format(bin_size if BINNING_ESTIMATOR else "kde"), dpi=300, bbox_inches="tight")
    plt.show()