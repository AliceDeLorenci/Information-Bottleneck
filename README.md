# Information-Bottleneck

Analysis of the application of the Information Bottleneck principle to Deep Neural Networks (DNN) based on Shwartz-Ziv, R. and Tishby, N., “Opening the Black Box of Deep Neural Networks via Information”.

## Code

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

        mi_xt_ty: compute mutual information (MI), using the binning estimator, between neural network inputs and layer activations, I(X,T), and between layer activations and targets, I(T, Y).

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

        save_weights: save the current weights of the model for the given epoch.
        """

- setups.py

        """
        Contains the setup dictionaries for the different experiments.

        setup_lookup: returns the desired setup dictionary.
        """

- utils.py

        """
        Utility functions for saving and loading data.

        temporizer: determines whether to skip saving for this epoch.

        save_setup: save the setup to a json file.

        load_setup: load the setup from a json file and convert lambda source code to lambda function.
        """ 

- argparser.py

        """
        Utility functions to parse command line arguments.

        get_main_parser: parse command line arguments and returns them as dictionary.

Experiments:

- TRAIN.py

        """
        python3 TRAIN.py -h

        Trains the NN defined by the chosen setup and saves the weights (for each epoch)
        in a separate file inside a timestamped subdirectory.
        """"

- MI.py

        """
        python3 MI.py -h

        Computes the I(X, T) and I(T, Y) for all layers and epochs and saves.
        """"
