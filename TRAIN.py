# python3 TRAIN.py <setup_idx>

# TODO: save initial random weights (epoch 0)!!
# TODO: save weights for each mini-batch

import sys
import os
import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from dataset import buildDatasets, buildDataLoader, loadMNISTData, loadSyntheticData
from mi import compute_mi, plot_info_plan
from nn import Network, train, test, save_activations, save_weights
from setups import setup_lookup
from utils import save_setup, load_setup

if __name__ == "__main__":

    setup_idx = int( sys.argv[1] )
    verbose = 1
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
    print("Weights will be saved in:", path)

    # Load dataset
    ratio = setup["train_ratio"]  # ratio of the training set to the test set
    if setup["dataset"] == "mnist":
        dataset = buildDatasets( *loadMNISTData(root="data"), ratio=ratio, name="mnist" )
    elif setup["dataset"] == "synthetic":
        dataset = buildDatasets( *loadSyntheticData(file="data/synthetic/var_u.mat"), ratio=ratio, name="synthetic" )
    else:
        raise ValueError("Unknown dataset")
    print("Dataset: {} -- nb training samples: {} -- nb test samples: {}".format(dataset["name"], len(dataset["train"]), len(dataset["test"])))

    loader = buildDataLoader(dataset, batch_size=setup["batch_size"])

    # Choose torch device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")     
    print("Using device:", device)

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

    train_loss = []
    test_loss = []
    test_acc = []
    for epoch in range(1, setup["n_epochs"] + 1):
            train_loss_item = train(model, setup, loader["train"], optimizer, device, epoch, verbose=verbose)
            test_loss_item, test_acc_item = test(model, setup, loader["test"], device, verbose=verbose)
            if save and epoch%save_interval == 0:
                    # save_activations(model, dataset["full"], epoch, device, path=path)
                    save_weights(model, epoch, path=path)
            train_loss.append(train_loss_item)
            test_loss.append(test_loss_item)
            test_acc.append(test_acc_item)

    plt.figure()
    plt.plot(train_loss, label="train")
    plt.plot(test_loss, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path+"loss.png", dpi=300, bbox_inches="tight")
    plt.show()

    np.savez_compressed( path+"loss", train_loss=train_loss, test_loss=test_loss, test_acc=test_acc)

    plt.figure()
    plt.plot(test_acc, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.savefig(path+"acc.png", dpi=300, bbox_inches="tight")
    plt.show()

    # mi_xt_epochs, mi_ty_epochs, epochs = compute_mi(dataset["full"], 
    #                                                 setup, 
    #                                                 path, 
    #                                                 interval=1, 
    #                                                 device=device)
    # np.savez_compressed( path+"mi", mi_xt_epochs=mi_xt_epochs, mi_ty_epochs=mi_ty_epochs, epochs=epochs)

    # plot_info_plan(mi_xt_epochs, mi_ty_epochs, epochs)
    # plt.savefig(path+"info-plan.png", dpi=300, bbox_inches="tight")
    # plt.show()