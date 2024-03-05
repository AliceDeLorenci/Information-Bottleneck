import argparse


def get_mi_parser():
    parser = argparse.ArgumentParser(description="Computes the I(X, T) and I(T, Y) for all layers and epochs and saves")

    parser.add_argument('--setup_idx', type=int, default=0, help='Index of the setup to use')
    parser.add_argument('--subdir', type=str, default=None, help='Tag of the subdirectory where activations/weights, subdirectory name is activations-<subdir_tag>')

    parser.add_argument('--data', type=str, default='full', choices=['full', 'test'], help='Dataset split to use for MI computation')
    parser.add_argument('--estimator', type=str, default='kde', choices=['kde', 'binning'], help='Estimator to use for MI computation')
    parser.add_argument('--bin_size', type=float, default=0.6, help='Bin size for binning estimator')
    parser.add_argument('--noise_variance', type=float, default=1e-3, help='Noise variance for KDE estimator')
    parser.add_argument('--temporize', action='store_true', help='Whether to temporize MI computation')

    return parser.parse_args()

def get_train_parser():
    parser = argparse.ArgumentParser(description="Trains the NN defined by the chosen setup and saves the weights (for each epoch)in a separate file inside a unique subdirectory")

    parser.add_argument('--shuffle', action='store_true', help='Whether to shuffle the data before splitting')
    parser.add_argument('--warmup', action='store_true', help='Whether to wait for network to start learning to start counting epochs')
    parser.add_argument('--threshold', type=float, default=0.69, help='Threshold for the warmup')
    parser.add_argument('--seed', type=int, default=None, help='Seed for all random number generators')
    parser.add_argument('--setup_idx', type=int, default=0, help='Index of the setup to use')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    parser.add_argument('--subdir', type=int, default=None, help='Subdirectory to save the activations/weights')
    parser.add_argument('--temporize', action='store_true', help='Whether to temporize storing of activations/weights')

    """
    parser.add_argument('--nn_type', type=str, default='mlp', choices=['mlp', 'cnn'], help='Type of neural network to use')
    parser.add_argument('--nout', type=int, default=1, choices=[1, 10], help='Number of outputs')
    parser.add_argument('--nlayers', type=int, default=2, help='Number of hidden layers in the networks')

    # CNN specifications
    parser.add_argument('--nin_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--kernel_size', type=int, default=3, help='Size of the convolutional kernel')
    parser.add_argument('--nfilters', type=int, default=16, help='Number of filters in the convolutional layers')

    # MLP specifications
    parser.add_argument('--nin', type=int, default=784, help='Input dimension')
    parser.add_argument('--nhid', type=int, default=600, help='Number of hidden units in a hidden layer')
    
    # Initial training specifications
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for SGD optimization')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for SGD optimization')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for SGD optimization')
    parser.add_argument('--weight_decay', type=float, default=0., help='Weight decay for SGD optimization')
    
    # PAC-Bayes bound optimization specifications
    parser.add_argument('--lr2', type=float, default=0.001, help='Learning rate for PAC-Bayes bound optimization')
    parser.add_argument('--sigma_init', type=float, default=1., help='Scaling to apply to the initial value of s')  # 1 for true-labels, 0.1 for random-labels
    parser.add_argument('--T', type=int, default=5000, help='Number of iterations for PAC-Bayes bound optimization') # paper uses 200 000
    parser.add_argument('--nb_snns', type=int, default=200, help='Number of SNNs to sample for MC approximation') # paper uses 150 000
    parser.add_argument('--best_loss_patience', type=int, default=1000, help='Patience of 2nd loop best loss')
    
    # Scheduler specifications
    parser.add_argument('--warmup_pct', type=float, default=0.1, help='Percentage of iterations to warm up')
    
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    """

    return parser.parse_args() 