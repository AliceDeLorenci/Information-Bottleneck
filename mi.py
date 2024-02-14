import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

def get_label_distribution(target):
    """
    Compute the distribution of the labels in the dataset.

    Args:
        target: torch.tensor, targets for the input data (number of samples)
    """

    count = F.one_hot(target.to(torch.int64)).sum(dim=0)
    return count/len(target)

def get_distribution(x): # compute array's row distribution

    """
    unique, inverse_indices, count = np.unique(x, axis=0, return_inverse=True, return_counts=True)
    unique : sorted unique values
    inverse_indices : indices to reconstruct original array from unique array
    count : number of times each unique value appears in the original array

    obs: the unique_inverse indices will be used instead of the unique values since what matters is the distribution, not the values themselves
    """
    _, inverse_indices, count = np.unique(x, axis=0, return_inverse=True, return_counts=True)
    return count / np.sum(count), inverse_indices

def mi_xt_ty(x, y, t, p_y, n_bins=30, bounds=[-1,1]):
    """
    Compute mutual information (MI) between neural network inputs and layer activations and between layer activations and targets.

    Args:
        x : inputs
        y : targets
        t : activations
        p_y : distribution of the targets
        n_bins : number of bins used to discretize activations
        bounds : bounds for the activation values
    """

    # discretize activations
    bin_size = (bounds[1] - bounds[0]) / n_bins
    t_binned = (t - bounds[0]) // bin_size          # determine to which bin each activation value belongs to and substitute for binned value

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

def compute_mi(dataset, folder, interval=100, path="./save/"):
    activation_file_name = []
    for file in os.listdir(path+folder):
        if file.startswith("activations_epoch_"):
            activation_file_name.append(file)

    p_y = get_label_distribution(dataset.targets)

    mi_xt_epochs = []
    mi_ty_epochs = []
    epochs = []
    
    counter = 0
    for file in activation_file_name:
        counter += 1
        print("{}/{}".format(counter, len(activation_file_name)), end='\r')
        
        epoch = int(file.split('_')[-1].split('.')[0])
        if epoch % interval != 0:
            continue

        activations = np.load(path+folder+"/"+file)
        mi_xt_layers = []
        mi_ty_layers = []
        for key in activations:
            mi_xt, mi_ty = mi_xt_ty(dataset.data, dataset.targets, activations[key], p_y)
            mi_xt_layers.append( mi_xt )
            mi_ty_layers.append( mi_ty )

        mi_xt_epochs.append( mi_xt_layers )
        mi_ty_epochs.append( mi_ty_layers )
        epochs.append( epoch )
    
    return mi_xt_epochs, mi_ty_epochs, epochs

def plot_info_plan(mi_xt, mi_yt, epochs):
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

    n_saved_epochs = mi_xt.shape[0]
    n_layers = mi_xt.shape[1]

    plt.figure()
    
    for e in range(n_saved_epochs):
        plt.plot(mi_xt[e, :], mi_yt[e, :], c='k', linewidth=0.5, alpha=0.5)
    for l in range(n_layers):
        scatter = plt.scatter(mi_xt[:, l], mi_yt[:, l], c=epochs, cmap='gnuplot', s=20, zorder=3)

    cb = plt.colorbar(scatter, ticks=[epochs[0], epochs[-1]])
    cb.ax.set_title('Epochs', fontsize=10)
    plt.show()