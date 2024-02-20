import sys
import os
import datetime
from pathlib import Path

import torch

from dataset import *
from mi import *   
from nn import *
from setups import *
from utils import *

# Usage: python3 compute_activations.py <setup_idx> <verbose> <folder> 
# <setup_idx> : int, index of the setup to use, retrieved by setups.py
# <verbose> : int, verbosity level (default: 0)
# <folder> : str, folder in which to save the activations (default: setup-<setup_idx>)

# if main
if __name__ == "__main__":
    
    # Command line arguments
    setup_idx = int( sys.argv[1] )

    if len(sys.argv) > 2:
        verbose = int( sys.argv[2] )
    else:
        verbose = 0
    
    if len(sys.argv) > 3:
        dir = sys.argv[3]
    else:
        dir = "setup-{}".format(setup_idx)

    # Load setup
    setup = setup_lookup(setup_idx)

    # Directory to save the activations (create the directory if it does not exist)
    Path( "./"+dir ).mkdir(parents=True, exist_ok=True) 
    
    # The activations will be saved in a subdirectory unique to this execution
    while True:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        subdir = "activations-{}".format(timestamp)
        try:
            os.mkdir("./"+dir+"/"+subdir)
            break
        except:
             pass
    path = "./"+dir+"/"+subdir+"/"

    # Load dataset
    ratio = setup["train_ratio"]  # ratio of the training set to the test set
    if setup["dataset"] == "mnist":
        dataset = buildDatasets( *loadMNISTData(root="data"), ratio=ratio, name="mnist" )
    elif setup["dataset"] == "synthetic":
        dataset = buildDatasets( *loadSyntheticData(file="data/synthetic/var_u.mat"), ratio=ratio, name="synthetic" )
    
    loader = buildDataLoader(dataset, batch_size=setup["batch_size"])

    # Choose torch device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")     
    # print("Using device:", device)

    # Train model
    save = True
    save_interval = 1

    model = Network(
            input_dim=dataset["n_features"], 
            hidden_dims=setup["hidden_dims"],
            output_dim=setup["output_dim"],
            hidden_activation_f=setup["hidden_activation_f"],
            output_activation_f=setup["output_activation_f"]
            ).to(device)

    optimizer = setup["optimizer"]( model.parameters() )

    if save:
       save_setup(setup, path=path, fname="setup")

    for epoch in range(1, setup["n_epochs"] + 1):
            train(model, setup, loader["train"], optimizer, device, epoch, verbose=verbose)
            test(model, setup, loader["test"], device)
            if save and epoch%save_interval == 0:
                    save_activations(model, dataset["full"], epoch, device, path=path)
