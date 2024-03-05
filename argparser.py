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

    return parser.parse_args() 