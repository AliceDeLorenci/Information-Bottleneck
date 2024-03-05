# python3 MI.py -h

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
from argparser import get_mi_parser

if __name__ == "__main__":

    args = get_mi_parser()
    print("Arguments:", args.__dict__)

    setup_idx = args.setup_idx

    dir = "setup-{}".format(setup_idx)
    subdir = "activations-{}".format(args.subdir)
    path = "./"+dir+"/"+subdir+"/"
    assert os.path.exists(path), "Folder does not exist: {}".format(path)
    print('Path:', path)

    setup = load_setup(path)

    BINNING_ESTIMATOR = (args.estimator == 'binning')

    # Choose torch device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")     
    print("Using device:", device)

    # Load dataset
    ratio = setup["train_ratio"]  # ratio of the training set to the test set
    if setup["dataset"] == "mnist":
        dataset = buildDatasets( *loadMNISTData(root="data"), ratio=ratio, name="mnist", shuffle=setup["shuffle"], seed=setup["seed"] )
    elif setup["dataset"] == "synthetic":
        dataset = buildDatasets( *loadSyntheticData(file="data/synthetic/var_u.mat"), ratio=ratio, name="synthetic", shuffle=setup["shuffle"], seed=setup["seed"] )

    mi_xt_epochs, mi_ty_epochs, mi_xt_lb_epochs, mi_ty_lb_epochs, epochs = compute_mi(dataset[args.data], 
                                                                        setup,
                                                                        path, 
                                                                        bin_size=args.bin_size,
                                                                        noise_variance=args.noise_variance,
                                                                        device=device,
                                                                        binning_estimator=BINNING_ESTIMATOR,
                                                                        temporize=args.temporize)
    
    np.savez_compressed( path+"mi-{}".format(args.bin_size if BINNING_ESTIMATOR else "kde-{}".format(args.noise_variance)), mi_xt_epochs=mi_xt_epochs, mi_ty_epochs=mi_ty_epochs, epochs=epochs, mi_xt_lb_epochs=mi_xt_lb_epochs, mi_ty_lb_epochs=mi_ty_lb_epochs)

    plot_info_plan(mi_xt_epochs, mi_ty_epochs, epochs)
    plt.savefig(path+"info-plan-{}.png".format(args.bin_size if BINNING_ESTIMATOR else "kde-{}".format(args.noise_variance)), dpi=300, bbox_inches="tight")
    plt.show()