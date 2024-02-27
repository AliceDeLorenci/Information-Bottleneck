import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

import scipy as sp
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    """
    Encapsulates a dataset with standard access to the data and targets.

    Attributes:
        data: torch.tensor, input data.
        targets: torch.tensor, targets for the input data.
    
    Methods:
        __init__: constructor for the class.
        __len__: returns the number of samples in the dataset.
        __getitem__: returns the data and target for the given index.
    """
    def __init__(self, data, targets):
        super(CustomDataset, self).__init__()
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
def loadSyntheticData(file, dtype=torch.float32):
    """
    Load synthetic dataset used by Tishby et al. (2017). Store in torch tensor.

    Args:
        file: str, path for the dataset.
        dtype: torch.dtype, data type for the tensors.

    Returns:
        data: torch.tensor, input data (number of samples, number of features)
        targets: torch.tensor, targets for the input data (number of samples)
    """

    synthetic = sp.io.loadmat(file)

    data = torch.tensor(synthetic['F'], dtype=dtype)
    targets = torch.tensor(synthetic['y'].ravel())

    return data, targets

def loadMNISTData(root):
    """
    Load MNIST dataset from torchvision.datasets.MNIST.

    Args:
        root: str, root directory for storing the dataset.

    Returns:
        data: torch.tensor, input data (number of samples, number of features)
        targets: torch.tensor, targets for the input data (number of samples)
    """

    mnist_train = torchvision.datasets.MNIST(root=root, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(root=root, train=False, download=True)

    data = torch.vstack( [mnist_train.data, mnist_test.data] ).reshape(-1, 28*28).to(torch.float32) / 255.0
    targets = torch.hstack( [mnist_train.targets, mnist_test.targets] )

    return data, targets

def buildDatasets(data, targets, ratio=0.8, seed=None, name=None):
    """
    Build training and test datasets from the given data and targets and stores them in dictionary:
        dataset["train"]: training dataset.
        dataset["test"]: test dataset.
        dataset["full"]: full dataset.
        dataset["n_features"]: number of features in the dataset.
        dataset["name"]: name of the dataset.

    Args:
        data: torch.tensor, input data.
        targets: torch.tensor, targets for the input data.
        ratio: float, ratio of the training set to the test set.
        seed: int, random seed for the data shuffling.
        name: str, name of the dataset.
    
    Returns:
        dataset: dict, dictionary containing training and test datasets, as well as the full dataset and the number of features.
    """
    if seed is not None:
        torch.manual_seed(seed)

    n = len( data )
    n_train = int( n * ratio )
    idx = torch.randperm( n )

    dataset = dict()
    dataset["train"] = CustomDataset( data[ idx[:n_train] ], targets[ idx[:n_train] ] )
    dataset["test"] = CustomDataset( data[ idx[n_train:] ], targets[ idx[n_train:] ] )
    dataset["full"] = CustomDataset( data, targets )
    dataset["n_features"] = data.shape[1]
    dataset["name"] = name

    return dataset

def buildDataLoader(dataset, batch_size=64):
    """
    Build data loaders for the given dataset.
        loader["train"]: training data loader.
        loader["test"]: test data loader.
        loader["dataset"]: dictionary containing the dataset.

    Args:
        dataset: dict, dictionary containing at least the training and test datasets.
        batch_size: int, size of the mini-batches.
    
    Returns:
        loader: dict, dictionary containing training and test data loaders, as well as the dataset dictionary.
    """
    
    if batch_size is None: # full batch instead of mini-batch
        batch_size_train  = len( dataset["train"] )
        batch_size_test = len( dataset["test"] )
    else:
        batch_size_train = batch_size
        batch_size_test = batch_size

    loader = dict()
    loader["train"] = DataLoader( dataset["train"], batch_size=batch_size_train, shuffle=True )
    loader["test"] = DataLoader( dataset["test"], batch_size=batch_size_test, shuffle=True )
    loader["dataset"] = dataset

    return loader

def plotimg(arr, width=28, height=28):
        """
        Plot gray scale image given as array.

        Example usage in subplots:
            plt.figure()
            plt.subplot(2,2,1)
            data.plot(0)
            plt.subplot(2,2,2)
            data.plot(1)
            plt.show()
        """
        plt.imshow(arr.reshape(width, height), cmap='gray')