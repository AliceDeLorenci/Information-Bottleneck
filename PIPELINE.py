# python3 TRAIN.py -h

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
from miestimation import compute_mi, plot_info_plan, mi_kde_xt_ty, mi_xt_ty, get_label_distribution
from nn import Network, train, test, save_activations, save_weights, return_activations
from setups import setup_lookup
from utils import save_setup, load_setup, temporizer
from argparser import get_pipeline_parser

SAVE_ACTIVATIONS = False    # whether to save activations (True) or weights (False)

if __name__ == "__main__":

    args = get_pipeline_parser()

    if args.seed is None:
        args.seed = np.random.randint(0, 1000)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    verbose = args.verbose
    temporize = args.temporize  # whether to temporize storing of activations/weights

    # Load setup
    setup_idx = args.setup_idx
    setup = setup_lookup(setup_idx)
    print("Setup:", setup_idx)

    # Directory to save the activations/weights (create the directory if it does not exist)
    dir = "setup-{}".format(setup_idx)
    Path( "./"+dir ).mkdir(parents=True, exist_ok=True) 

    # The activations/weights will be saved in a subdirectory unique to this execution
    if args.subdir is not None:
        subdir = "activations-{}".format(args.subdir)
        path = "./"+dir+"/"+subdir+"/"
        try:
            os.mkdir(path)
        except:
            raise ValueError("The subdirectory already exists")
    else:
        while True:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            subdir = "activations-{}".format(timestamp)
            path = "./"+dir+"/"+subdir+"/"
            try:
                os.mkdir(path)
                break
            except:
                pass
    print("Results will be saved in:", path)

    # Save setup
    setup.update(args.__dict__)
    save_setup(setup, path=path, fname="setup")

    # Load dataset
    ratio = setup["train_ratio"]  # ratio of the training set to the test set
    if setup["dataset"] == "mnist":
        dataset = buildDatasets( *loadMNISTData(root="data"), ratio=ratio, name="mnist", shuffle=args.shuffle, seed=args.seed )
    elif setup["dataset"] == "synthetic":
        dataset = buildDatasets( *loadSyntheticData(file="data/synthetic/var_u.mat"), ratio=ratio, name="synthetic", shuffle=args.shuffle, seed=args.seed )
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
    model = Network(
            input_dim=dataset["n_features"], 
            hidden_dims=setup["hidden_dims"],
            output_dim=setup["output_dim"],
            hidden_activation=setup["hidden_activation"],
            output_activation=setup["output_activation"]
            ).to(device)

    optimizer = setup["optimizer"]( model.parameters() )


    train_loss = [] 
    test_loss = []  
    test_acc = []   

    mi_xt_epochs = []
    mi_ty_epochs = []
    # mi_xt_lb_epochs = []
    # mi_ty_lb_epochs = []
    epochs = []
    bin_size = args.bin_size
    noise_variance = args.noise_variance
    binning_estimator = (args.estimator == 'binning')
    p_y = get_label_distribution(dataset[args.data].targets)

    # Save initial random weights
    epoch = 0
    # if SAVE_ACTIVATIONS:
    #     save_activations(model, dataset["full"], epoch, device, path=path)
    # else:
    #     save_weights(model, epoch, path=path)
    
    epoch_is_iteration = setup["epoch_is_iteration"] if "epoch_is_iteration" in setup else False

    # for epoch in range(1, setup["n_epochs"] + 1):
    ### BEGIN !!!
    epoch = 0 
    count = 0 
    learning = False 
    learning_count = 1 
    while True:   
        epoch += 1  
        if not learning: 
            count += 1 
        else: 
            learning_count += 1 

        if count > setup["n_epochs"] or learning_count > setup["n_epochs"]: 
            break 
    ### END !!!

        train_loss_item = train(model, setup, loader["train"], optimizer, device, epoch, verbose=verbose, epoch_is_iteration=epoch_is_iteration)
        test_loss_item, test_acc_item = test(model, setup, loader["test"], device, verbose=verbose)

        ### BEGIN !!!
        if args.warmup and test_loss_item <= args.threshold: 
            if not learning: 
                print( "== {} started learning ({}) ==".format(args.subdir, count) ) 
            learning = True 
        ### END !!!


        train_loss.append(train_loss_item)
        test_loss.append(test_loss_item)
        test_acc.append(test_acc_item)

        if temporize:
            temporize_flag = temporizer(epoch)
            if temporize_flag:
                continue
            else:
                pass

        activations = return_activations(model, dataset[args.data], device)  

        mi_xt_layers = []
        mi_ty_layers = []
        mi_xt_lb_layers = []
        mi_ty_lb_layers = []

        # activations for hidden layers
        N_LAYERS = len(activations)
        for l, act in enumerate(activations):
            if binning_estimator:
                if l == N_LAYERS-1:
                    activation_type = setup["output_activation"]
                else:
                    activation_type = setup["hidden_activation"]
                mi_xt, mi_ty = mi_xt_ty(dataset[args.data].data, dataset[args.data].targets, act if SAVE_ACTIVATIONS else act.cpu().numpy(), p_y, activation=activation_type, bin_size=bin_size)
                mi_xt_lb = mi_ty_lb = None
            else:
                mi_xt, mi_ty = mi_kde_xt_ty(dataset[args.data].data, dataset[args.data].targets, act if not SAVE_ACTIVATIONS else torch.from_numpy(act).to(device), p_y, noise_variance=noise_variance)[:2]
                mi_xt = mi_xt.cpu().numpy()
                mi_ty = mi_ty.cpu().numpy()
                # mi_xt_lb = mi_xt_lb.cpu().numpy()
                # mi_ty_lb = mi_ty_lb.cpu().numpy()

            mi_xt_layers.append( mi_xt )
            mi_ty_layers.append( mi_ty )
            # mi_xt_lb_layers.append( mi_xt_lb )
            # mi_ty_lb_layers.append( mi_ty_lb ) 
        
        mi_xt_epochs.append( mi_xt_layers )
        mi_ty_epochs.append( mi_ty_layers )
        # mi_xt_lb_epochs.append( mi_xt_lb_layers )
        # mi_ty_lb_epochs.append( mi_ty_lb_layers )
        epochs.append( epoch )
        
        # if SAVE_ACTIVATIONS:
        #     save_activations(model, dataset["full"], epoch, device, path=path)
        # else:
        #     save_weights(model, epoch, path=path)


    np.savez_compressed( path+"mi-{}".format(args.bin_size if binning_estimator else "kde-{}".format(args.noise_variance)), mi_xt_epochs=mi_xt_epochs, mi_ty_epochs=mi_ty_epochs, epochs=epochs ) # , mi_xt_lb_epochs=mi_xt_lb_epochs, mi_ty_lb_epochs=mi_ty_lb_epochs)

    plot_info_plan(mi_xt_epochs, mi_ty_epochs, epochs)
    plt.savefig(path+"info-plan-{}.png".format(args.bin_size if binning_estimator else "kde-{}".format(args.noise_variance)), dpi=300, bbox_inches="tight")
    plt.show()

    # Plot training and test loss
    plt.figure()
    plt.plot(train_loss, label="train")
    plt.plot(test_loss, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path+"loss.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Save training and test loss
    np.savez_compressed( path+"loss", train_loss=train_loss, test_loss=test_loss, test_acc=test_acc)

    # Plot test accuracy
    plt.figure()
    plt.plot(test_acc, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.savefig(path+"acc.png", dpi=300, bbox_inches="tight")
    plt.show()

    print('Setup:', setup_idx)
    print('Path:', path)

    # mi_xt_epochs, mi_ty_epochs, epochs = compute_mi(dataset["full"], 
    #                                                 setup, 
    #                                                 path, 
    #                                                 interval=1, 
    #                                                 device=device)
    # np.savez_compressed( path+"mi", mi_xt_epochs=mi_xt_epochs, mi_ty_epochs=mi_ty_epochs, epochs=epochs)

    # plot_info_plan(mi_xt_epochs, mi_ty_epochs, epochs)
    # plt.savefig(path+"info-plan.png", dpi=300, bbox_inches="tight")
    # plt.show()