# Information-Bottleneck

Analysis of the application of the Information Bottleneck principle to Deep Neural Networks (DNN) based on Shwartz-Ziv, R. and Tishby, N., “Opening the Black Box of Deep Neural Networks via Information”.

## Code

### Packages

    numpy-1.26.4
    torch-2.2.0
    torchvision-0.17.0
    matplotlib-3.8.2
    scipy-1.12.0

### Files 

Modules:
- dataset.py

    """
    Support methods and classes for loading and handling datasets.

    class CustomDataset: encapsulates a dataset with standard access to the data and targets.

    loadSyntheticData: load synthetic dataset used by Tishby et al. (2017).

    loadMNISTData: load MNIST dataset from torchvision.datasets.MNIST.

    buildDatasets: build training and test datasets from the given data and targets.

    buildDataLoader: build data loaders (train and test) for the given dataset.

    plotimg: plot gray scale image given as array.
    """

- mi.py

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

- nn.py

    """
    This module contains the implementation of a simple feedforward neural network using PyTorch.

    class Network: implementation of a simple feedforward neural network using PyTorch.

    train: train the model for one epoch.

    test: evaluate the model on test set.

    save_activations: save the activations of the model for the given epoch and dataset.
    """

- setups.py

    """
    Contains the setup dictionaries for the different experiments.

    setup_lookup: returns the desired setup dictionary.
    """

- utils.py

    """
    Utility functions for saving and loading data.

    save_setup: save the setup to a json file.

    load_setup: load the setup from a json file and convert lambda source code to lambda function.
    """ 

Run experiments:
- IB.ipynb

- IB.py

    """
    python3 IB.py <setup_idx>

    1. Trains the NN defined by the chosen setup and saves the activations (for each epoch)
    in a separate file inside a timestamped subdirectory:
        setup-<setup_idx>/activations-<timestamp>/activations_epoch_<epoch_number>.npz
    2. Saves the losses inside the timestamped subdirectory, as well as loss and accuracy plots:
        setup-<setup_idx>/activations-<timestamp>/loss.npz
    3. Computes the I(X, T) and I(T, Y) for all layers and epochs and saves:
        setup-<setup_idx>/activations-<timestamp>/mi.npz
    """"

- MI_binning.py

    """
    python3 MI_binning.py setup-<setup_idx>/activations-<timestamp>/ <bin_size>

    1. Computes the I(X, T) and I(T, Y) for all layers and epochs and saves:
        setup-<setup_idx>/activations-<timestamp>/mi-<bin_size>.npz

    The folder setup-<setup_idx>/activations-<timestamp>/ should contain the activations, for each epoch, in separate folders:
        setup-<setup_idx>/activations-<timestamp>/activations_epoch_<epoch_number>.npz
    """"

## TODO

- Synthetic dataset:
    - [ok] tanh 
        - Motivation: reproducing paper
        - setup 1
    - [ok] relu 
        - Motivation: very restricted test setting in the paper, testing more popular activation
        - setup 2
    - [] what is the impact o regularization: weight decay (setup 3 and 4)
        - Motivation: compression phase is hypothetised to relate to generalization, it is well known regularization propotes regularization, therefore the hypothesis is that we will observe more significant compression
        - setup 3 (0.0001)
        - setup 4 (0.001)
    - [] what is the impact of the MI estimator
        - [] test different bin sizes
        - [] test new MI estimator (Kraskov, 2004)
        - [] test new MI estimator (Kolchinsky, 2017)
    - [] test other bounded activations: sigmoid
    - [] test other unbounded activation: leaky relu, silu

- Reproduce results from IB paper
- Extend to MNIST: tanh, relu
- Check discussion on https://openreview.net/forum?id=ry_WPG-A-
- Better MI estimators due to Kraskov 2003, Kolchinsky 2017 and Goldfeld 2019
- Regularization

## Related work

### Estimating Information Flow in Deep Neural Networks, 2019

Binning-based MI estimation approaches:
- attractive because of computational efficiency
- fluctuations of MI might be due to estimation errors rather then changes in MI

Deterministic NN with continuous non-linearities:
- if Px is continuous, then I(X, T) = infty
- if Px is discrete, then I(X, T) = H(X) = constant
- this is a consequence of the fact that deterministic DNNs can encode information about X in arbitraly fine variations of T

Contribution: stochastic NN framework
- I(X, T) reflects DNN true operating conditions

Contribution: MI estimator

## References

[Example project](https://github.com/fournierlouis/synaptic_sampling_rbm/blob/master/Rapport_Projet_Neurosciences___Synaptic_Sampling.pdf)

- Shwartz-Ziv & Tishby, 2017
    - HTML version: https://ar5iv.labs.arxiv.org/html/1703.00810
    - Talk: https://www.youtube.com/watch?v=bLqJHjXihK8
    - News article: https://www.quantamagazine.org/new-theory-cracks-open-the-black-box-of-deep-learning-20170921/#
    - Code: https://github.com/ravidziv/IDNNs
    - Horrible code, unsupported

- Andrew Michael Saxe, Yamini Bansal, Joel Dapello, Madhu Advani, Artemy Kolchinsky, Brendan Daniel Tracey, David Daniel Cox, On the Information Bottleneck Theory of Deep Learning, ICLR 2018.
    - They criticize the IB principle
    - Code: https://github.com/artemyk/ibsgd
    - Nice code but doesn't work 
        - _feed_targets: https://stackoverflow.com/questions/51140950/how-to-obtain-the-gradients-in-keras
    - Might want to check the way they compute MI

- Nice implementation of Tishby's paper
    - https://github.com/shalomma/PytorchBottleneck
    - Pytorch
    - Works
    - Test with latest version of Pytorch

- Simple implementation of Tishby's paper
    - https://github.com/stevenliuyi/information-bottleneck
    - Haven't tested it yet

- List of IB papers
    - https://github.com/ZIYU-DEEP/Awesome-Information-Bottleneck

# Micro-Project

Test Shwartz-Ziv & Tishby, 2017 ideas on real dataset (MNIST):
- binary classification
- multi-class
- do different regularizations impact compression phase?

# Related work

- Noam Slonim, Agglomerative Information Bottleneck, 1999
    - Bottom up version of the IB iterative algorithm
- Ravid Shwartz-Ziv, Information Flow in Deep Neural Networks, 2022
    - PhD thesis, sumarized many IB works
- Elad Schneidman, Analysing Neural Codes using the Information Bottleneck Method, 2001
<br><br><br>
- Andrew M. Saxe, **On the Information Bottleneck Theory of Deep Learning**, 2018
    - Criticize the IB principle (check how they estimate MI) [GitHub](https://github.com/artemyk/ibsgd)
- Ziv Goldfeld, **Estimating Information Flow in Deep Neural Networks**, 2019
    - Accurate estimation of IB in DNN
    - Takes issue with Opening the Black Box of Deep Neural Networks via Information
<br><br><br>
- Alexander A. Alemi, Deep Variational Information Bottleneck, 2019
    - IB as an objective function [GitHub](https://github.com/alexalemi/vib_demo) [Implementation 1](https://github.com/udeepam/vib) [Implementation 2](https://github.com/1Konny/VIB-pytorch)
- Zoe Piran, The Dual Information Bottleneck, 2020
    - Addresses shortcomings of the IB framework, dualIB as an objective function [GitHub](https://github.com/ravidziv/dual_IB)