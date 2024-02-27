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

###
def get_dists(X):
    """Keras code to compute the pairwise distance matrix for a set of
    vectors specifie by the matrix X.
    """
    x2 = torch.unsqueeze(torch.sum(torch.square(X), dim=1), dim=1)
    dists = x2 + torch.transpose(x2, 0, 1) - 2*torch.matmul(X, torch.transpose(X, 0, 1))
    return dists

def get_shape(x):
    dims = float( x.size()[1] )
    N    = float( x.size()[0] )
    return dims, N

def entropy_estimator_kl(x, var):
    # KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I 
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    #  and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
    dims, N = get_shape(x)
    dists = get_dists(x)
    dists2 = dists / (2*var)
    normconst = (dims/2.0)*np.log(2*np.pi*var)
    lprobs = torch.logsumexp(-dists2, axis=1) - np.log(N) - normconst
    h = -torch.mean(lprobs)
    return dims/2 + h

def entropy_estimator_bd(x, var):
    # Bhattacharyya-based lower bound on entropy of mixture of Gaussians with covariance matrix var * I 
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    dims, N = get_shape(x)
    val = entropy_estimator_kl(x,4*var)
    return val + torch.log(0.25)*dims/2

def kde_condentropy(output, var):
    # Return entropy of a multivariate Gaussian, in nats
    dims = output.size()[1]
    return (dims/2.0)*(np.log(2*np.pi*var) + 1)

def mi_kde_xt_ty(x, y, t, p_y):
    """
    Compute mutual information (MI), using the KDE estimator, between neural network inputs and layer activations, I(X,T), and between layer activations and targets, I(T, Y).

    Args:
        x : inputs
        y : targets
        t : activations
        p_y : distribution of the targets
    """
    DO_LOWER = False

    # Compute marginal entropies
    # noise_variance = 1e-3 # synthetic
    noise_variance = 1e-1   # mnist
    h_upper = entropy_estimator_kl(t, noise_variance)
    if DO_LOWER:
        h_lower = entropy_estimator_bd(t, noise_variance)
        
    # Layer activity given input. This is simply the entropy of the Gaussian noise
    hM_given_X = kde_condentropy(t, noise_variance)

    # Compute conditional entropies of layer activity given output
    NUM_LABELS = len(p_y)
    hM_given_Y_upper=0.
    for i in range(NUM_LABELS):
        idx = y==i
        hcond_upper = entropy_estimator_kl(t[idx,:], noise_variance) 
        hM_given_Y_upper += p_y[i] * hcond_upper

    if DO_LOWER:
        hM_given_Y_lower=0.
        for i in range(NUM_LABELS):
            hcond_lower = entropy_estimator_bd(t[idx,:], noise_variance)
            hM_given_Y_lower += p_y[i] * hcond_lower

    nats2bits = 1.0/np.log(2)
    mi_xt = nats2bits * (h_upper - hM_given_X)
    mi_ty = nats2bits * (h_upper - hM_given_Y_upper)
    h_t_upper = nats2bits * h_upper

    mi_xt_lower = mi_ty_lower = h_t_lower = None
    if DO_LOWER:
        mi_xt_lower = nats2bits * (h_lower - hM_given_X)
        mi_ty_lower = nats2bits * (h_lower - hM_given_Y_lower)
        h_t_lower = nats2bits * h_lower

    return mi_xt, mi_ty, h_t_upper, mi_xt_lower, mi_ty_lower, h_t_lower

### 

def mi_xt_ty(x, y, t, p_y, activation="tanh", bin_size=0.05):
    """
    Compute mutual information (MI), using the binning estimator, between neural network inputs and layer activations, I(X,T), and between layer activations and targets, I(T, Y).

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

def compute_mi(dataset, setup, path, interval=100, bin_size=None, device=torch.device("cpu"), binning_estimator=False):
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
        bin_size : size of the bins used to discretize activations, relevant when using binnning estimator
        device : torch.device, device to use for the computation, relevant when the weights (and not the activations) were stored or when using KDE estimator
        binning_estimator : bool, whether to use binning estimator (True) or KDE (False) to compute the MI
    
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
        ACTIVATIONS = True
        file_name = activation_file_name
    else:
        ACTIVATIONS = False
        file_name = weights_file_name
    print("Data type:", type)
    
    # if the folder contains layer weights, load the model
    if not ACTIVATIONS:
        model = Network(
            input_dim=dataset.data.shape[1], 
            hidden_dims=setup["hidden_dims"],
            output_dim=setup["output_dim"],
            hidden_activation_f=setup["hidden_activation_f"],
            output_activation_f=setup["output_activation_f"]
            ).to(device)
    
    # empirical probability distributions of the targets
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

        if ACTIVATIONS:
            activations = np.load(path+file)                                    # load activations (numpy array)
            activations = [ activations[key] for key in activations.keys() ]
        else:
            load_weights(model, device, path+file)                              # load weights
            activations = return_activations(model, dataset, device)            # compute activations (torch tensor on device)

        mi_xt_layers = []
        mi_ty_layers = []

        # activations for hidden layers
        N_LAYERS = len(activations)
        for l, act in enumerate(activations):
            if binning_estimator:
                if l == N_LAYERS-1:
                    activation_type = setup["output_activation"]
                else:
                    activation_type = setup["hidden_activation"]
                mi_xt, mi_ty = mi_xt_ty(dataset.data, dataset.targets, act if ACTIVATIONS else act.cpu().numpy(), p_y, activation=activation_type, bin_size=bin_size)
            else:
                mi_xt, mi_ty = mi_kde_xt_ty(dataset.data, dataset.targets, act if not ACTIVATIONS else torch.from_numpy(act).to(device), p_y)[:2]
                mi_xt = mi_xt.cpu().numpy()
                mi_ty = mi_ty.cpu().numpy()
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