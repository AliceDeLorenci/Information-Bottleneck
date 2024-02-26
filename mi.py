"""
This file contains the functions to compute the mutual information 
between the input data and the layer activations, I(X,T), and between 
the layer activations and the targets, I(T,Y), and to plot the information plane.

get_label_distribution: compute the distribution of the labels in the dataset.

get_distribution: compute the row distribution of the given array.

mi_xt_ty: compute mutual information (MI) between neural network inputs and layer activations, I(X,T), 
and between layer activations and targets, I(T, Y).

compute_mi: load all activation data from folder and compute the mutual information.

plot_info_plan: plot the given mutual information values for each layer and each epoch in the information plane.
"""
import torch
import torch.nn.functional as F

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

from nn import Network, load_weights, return_activations

def get_label_distribution(target):
    """
    Compute the distribution of the labels in the dataset.

    Args:
        target: torch.tensor, targets for the input data (number of samples)
    """

    count = F.one_hot(target.to(torch.int64)).sum(dim=0)
    return count/len(target)

def get_distribution(x): # compute array's row distribution
    """"
    Compute the row distribution of the given array.
    
    Args:
        x: np.array, input array (number of samples, number of features)
    
    Observations:
        unique, inverse_indices, count = np.unique(x, axis=0, return_inverse=True, return_counts=True)
            unique : sorted unique values
            inverse_indices : indices to reconstruct original array from unique array
            count : number of times each unique value appears in the original array
        The unique_inverse indices will be used instead of the unique values since
        what matters is the distribution, not the values themselves.
    """
    _, inverse_indices, count = np.unique(x, axis=0, return_inverse=True, return_counts=True)
    return count / np.sum(count), inverse_indices

def mi_xt_ty(x, y, t, p_y, activation="tanh", bin_size=0.05):
    """
    Compute mutual information (MI) between neural network inputs and layer activations, I(X,T), and between layer activations and targets, I(T, Y).

    Args:
        x : inputs
        y : targets
        t : activations
        p_y : distribution of the targets
        activation : activation function used in the layer, default is "tanh", necessary to determine the lower bound for the activation values
        bin_size : size of the bins used to discretize activations
    """

    if activation == "tanh":
        lb = -1
    elif activation == "relu" or activation == "sigmoid":
        lb = 0
    else:
        lb = np.min(t)
        print("Activation function not supported, defaulting to np.min lower bound")
    
    if bin_size is None:
        bin_size = 0.05

    t_binned = (t - lb) // bin_size          # determine to which bin each activation value belongs to and substitute for binned value

    p_t, inverse_indices_t = get_distribution( t_binned )    # binned activation's distribution
    
    h_layer = -np.sum(p_t * np.log(p_t)) # H(T): entropy of activation's distribution

    # H(T|X) = 0 (the network is deterministic, i.e., the activations are fully determined by the inputs)
    # threrefore I(X;T) = H(T) - H(T|X) = H(T)
    mi_xt = h_layer

    # H(T|Y): entropy of the activations given the outputs
    h_layer_given_output = 0.
    for y_val in range( len(p_y) ):
        t_y = inverse_indices_t[ y == y_val ]
        _, count = np.unique(t_y, return_counts=True)
        p_t_given_y = count / np.sum(count)                                                       # binned activation's distribution conditionned on the outputs
        h_layer_given_output += - p_y[y_val] * np.sum(p_t_given_y * np.log(p_t_given_y))
    mi_ty = h_layer - h_layer_given_output

    return mi_xt, mi_ty

def compute_mi(dataset, setup, path, interval=100, bin_size=None, device=torch.device("cpu")):
    """
    If the given path stores activations, load all activation data from folder and compute the mutual information.
    If the given path stores weights, load all weights data, compute activations and compute the mutual information.

    The files in the folder should be named either:
        "activations_epoch_*.<npy/npz>"
    or:
        "weights_epoch_*.pt"
    
    In the first case, each file should contain the activations for all layers in the network for a given epoch, 
    and each array in the file should contain the activations for a given layer. In the second case, each file
    should contain the model state dictionary for a given epoch.

    Args:
        dataset : dataset, dataset to use to compute the MI
        setup : dictionary, setup dictionary containing at least the model configuration
        path : path to the folder containing the activation data
        interval : interval between epochs to compute the mutual information
        bin_size : size of the bins used to discretize activations, optional
    
    Returns:
        mi_xt_epochs : list of lists, mutual information between input and layer activations for each epoch (axis 0) and each layer (axis 1)
        mi_ty_epochs : list of lists, mutual information between layer activations and targets for each epoch (axis 0) and each layer (axis 1)
        epochs : epochs at which the mutual information was computed
    """
    activation_file_name = []
    for file in os.listdir(path):
        if file.startswith("activations_epoch_"):
            activation_file_name.append(file)
    
    weights_file_name = []
    for file in os.listdir(path):
        if file.startswith("weights_epoch_"):
            weights_file_name.append(file)

    if len(activation_file_name) > 0:
        type = "activations"
        file_name = activation_file_name
    else:
        type = "weights"
        file_name = weights_file_name
    print("Data type:", type)
    
    if type == "weights":
        model = Network(
            input_dim=dataset.data.shape[1], 
            hidden_dims=setup["hidden_dims"],
            output_dim=setup["output_dim"],
            hidden_activation_f=setup["hidden_activation_f"],
            output_activation_f=setup["output_activation_f"]
            ).to(device)
    
    # empirical probability of the targets
    p_y = get_label_distribution(dataset.targets)

    mi_xt_epochs = []
    mi_ty_epochs = []
    epochs = []
    
    counter = 0
    for file in file_name:
        counter += 1
        print("{}/{}".format(counter, len(file_name)), end='\r')
        
        epoch = int(file.split('_')[-1].split('.')[0])
        if epoch % interval != 0:
            continue

        if type == "activations":
            activations = np.load(path+file)
            activations = [ activations[key] for key in activations.keys() ]
        else:
            load_weights(model, device, path+file)
            activations = return_activations(model, dataset, device)

        mi_xt_layers = []
        mi_ty_layers = []

        # activations for hidden layers
        for act in activations[:-1]:
            activation_type = setup["hidden_activation"]
            mi_xt, mi_ty = mi_xt_ty(dataset.data, dataset.targets, act, p_y, activation=activation_type, bin_size=bin_size)
            mi_xt_layers.append( mi_xt )
            mi_ty_layers.append( mi_ty )

        # activations for output layer
        activation_type = setup["output_activation"]
        mi_xt, mi_ty = mi_xt_ty(dataset.data, dataset.targets, activations[-1], p_y, activation=activation_type, bin_size=bin_size)
        mi_xt_layers.append( mi_xt )
        mi_ty_layers.append( mi_ty )

        mi_xt_epochs.append( mi_xt_layers )
        mi_ty_epochs.append( mi_ty_layers )
        epochs.append( epoch )
    
    return mi_xt_epochs, mi_ty_epochs, epochs

def plot_info_plan(mi_xt, mi_yt, epochs, ticks=[], markup=None, max_epoch=None):
    """
    Plot the given mutual information values for each layer and each epoch in the information plane.

    Args:
        mi_xt : mutual information between input and layer activations
        mi_yt : mutual information between layer activations and targets
        epochs : epochs at which the mutual information was computed
    """

    mi_xt = np.array( mi_xt )
    mi_yt = np.array( mi_yt )
    epochs = np.array( epochs )

    sorted_idx = np.argsort(epochs)
    mi_xt = mi_xt[sorted_idx]
    mi_yt = mi_yt[sorted_idx]
    epochs = epochs[sorted_idx]

    if max_epoch is not None:
        idx = np.where(epochs <= max_epoch)
        mi_xt = mi_xt[idx]
        mi_yt = mi_yt[idx]
        epochs = epochs[idx]

    n_saved_epochs = mi_xt.shape[0]
    n_layers = mi_xt.shape[1]

    plt.figure()
    
    for l in range(n_layers):
        scatter = plt.scatter(mi_xt[:, l], mi_yt[:, l], c=epochs, cmap='gnuplot', s=30, zorder=3)
    cmap = mpl.cm.get_cmap('gnuplot', n_saved_epochs)
    color = cmap(epochs)

    for e in range(n_saved_epochs):
        plt.plot(mi_xt[e, :], mi_yt[e, :], c=color[e], linewidth=0.5, alpha=0.2)

    cb = plt.colorbar(scatter, ticks=[epochs[0], epochs[-1]])
    cb.ax.set_title('Epochs', fontsize=10)
    cb.set_ticks([epochs[0], epochs[-1]]+ticks)

    plt.xlabel('I(X;T)')
    plt.ylabel('I(T;Y)')

    if markup is not None:
        plt.scatter(mi_xt[markup, :], mi_yt[markup, :], c='green', s=30, zorder=3)
        plt.plot(mi_xt[markup, :], mi_yt[markup, :], color='green', linewidth=0.5)
    # plt.show()