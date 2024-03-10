import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import inspect
import datetime
import os
import numpy as np

"""
In order to have more generic code, the network and training parameters should be specified through the setup dictionary. The required keys are:
- **hidden_dims**: list of hidden layers sizes
- **output_dim**: output layer size
- **hidden_activation_f**: activation function for the hidden layers
- **output_activation_f**: activation function for the output layer
- **n_epochs**: number of epochs
- **batch_size**: batch size, if None, do batch training (use the total training data for each update) instead of mini-batch
- **optimizer**: optimizer from ```torch.optim``` encapsulated on a lambda function that receives as argument only the parameters to optimize 
- **loss_function**: loss function with the signature ```loss_function(output, target, reduction='mean')```
where output and target are the network output and the target values, respectively, 
the reduction parameter is necessary due to train and test time differences, the default value must be 'mean'
- **evaluate_correct**: function to evaluate the number of correct predictions, with the signature ```evaluate_correct(output, target)```
where output and target are the network output and the target values, respectively

**WARNING:** When specifying functions, encapsulate them in **lambda expressions** so that the setup can be saved to and loaded from json files!
"""

class Network(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, hidden_activation="tanh", output_activation="sigmoid"):
        """
        Args:
            input_dim: int, size of the input layer.
            hidden_dims: list of int, sizes of the hidden layers.
            output_dim: int, size of the output layer.
            hidden_activation_function: torch.nn.functional or torch.nn activation function, activation function for the hidden layers.
            output_activation_function: torch.nn.functional or torch.nn activation function, activation function for the output layer.
        """
        super(Network, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden_layers = len(hidden_dims)
        
        if hidden_activation=="tanh":
            self.hidden_activation_f = torch.nn.functional.tanh
        elif hidden_activation=="relu":
            self.hidden_activation_f = torch.nn.functional.relu
        else:
            raise ValueError("Unknown activation")

        if output_activation=="sigmoid":
            self.output_activation_f = torch.nn.functional.sigmoid
        elif output_activation=="log_softmax":
            self.output_activation_f = lambda input: torch.nn.functional.log_softmax( input, dim=1 )
        else:
            raise ValueError("Unknown activation")
        

        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for dim in hidden_dims + [output_dim]:
            self.layers.append( nn.Linear(prev_dim, dim) )
            prev_dim = dim
        
    
    def forward(self, x):
        activations = []
        for layer in self.layers[:-1]:
            x = self.hidden_activation_f( layer(x) )
            activations.append( x )
            
        x = self.output_activation_f( self.layers[-1](x) )
        activations.append( x )
        return x, activations
    
def train(model, setup, train_loader, optimizer, device, epoch, verbose=1, epoch_is_iteration=False):
    """
    Train the model for one [epoch] mini-batch.

    Args:
        model: torch.nn.Module, model to train.
        setup: dict, setup dictionary containing the loss function
        train_loader: torch.utils.data.DataLoader, data loader for the training set.
        optimizer: torch.optim, optimization algorithm.
        device: torch.device, device to use for the computation.
        epoch: int, current epoch number.
        verbose: int, level of verbosity, 0 for no output, 1 for output at the end of the epoch, 2 for output at each batch.
    """

    model.train()

    ### 
    if epoch_is_iteration:

        data, target = next(iter(train_loader))

        # move data to device
        data, target = data.to(device), target.to(device)

        # forward pass
        output, _ = model(data)
        loss = setup["loss_function"](output, target)

        # optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    else:

        for batch_idx, (data, target) in enumerate(train_loader):
            # move data to device
            data, target = data.to(device), target.to(device)

            # forward pass
            output, _ = model(data)
            loss = setup["loss_function"](output, target)

            # optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose==2:
                print('\rEpoch {}: Batch {}/{}: train loss {:.4f} '.format(epoch, batch_idx+1, len(train_loader), loss.item()), end='')
        ### END FOR
            
    if verbose==1:
        print('Epoch {}: train loss {:.4f} '.format(epoch, loss.item()), end='')
    elif verbose==0:
        print('Epoch {}: train loss {:.4f} '.format(epoch, loss.item()), end='\r')

    return loss.item()

def test(model, setup, test_loader, device, verbose=1):
    """
    Evaluate the model on test set.

    Args:
        model: torch.nn.Module, model to evaluate.
        setup: dict, dictionary containing the loss function and the evaluation function.
        test_loader: torch.utils.data.DataLoader, data loader for the test set.
        device: torch.device, device to use for the computation.
    """

    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # move data to device
            data, target = data.to(device), target.to(device)
            output, _ = model(data)

            test_loss += setup["loss_function"](output, target, reduction='sum')
            correct += setup["evaluate_correct"](output, target)

    test_loss /= len(test_loader.dataset)
    correct /= len(test_loader.dataset)

    if verbose:
        print( '-- test loss: {:.4f} -- test acc: {:.4f}'.format(test_loss.item(), correct.item()) )
    # elif verbose == 1:
    #     print( '-- test loss: {:.4f} -- test acc: {:.4f}'.format(test_loss.item(), correct.item()), end='\r' )
    
    return test_loss.item(), correct.item()


def save_activations(model, dataset, epoch, device, path="./save/"):
    """
    Save the activations of the model for the given dataset and epoch on the file:
        <path>activations_epoch_<epoch>.npz

    Args:
        model: torch.nn.Module, model to evaluate.
        dataset: dataset.CustomDataset, dataset to evaluate the model on.
        epoch: int, current epoch number.
        device: torch.device, device to use for the computation.
        path: str, path to the directory where the data is saved.
    """

    with torch.no_grad():

        data = dataset.data.to(device)
        _, layer_activations = model(data)

        layer_activations = [ activation.detach().cpu().numpy() for activation in layer_activations ]

        np.savez_compressed(path+"activations_epoch_"+str(epoch), *layer_activations)

def return_activations(model, dataset, device):
    """
    Returns the activations of the model for the given dataset.

    Args:
        model: torch.nn.Module, model to evaluate.
        dataset: dataset.CustomDataset, dataset to evaluate the model on.
        device: torch.device, device to use for the computation.
    """
    with torch.no_grad():

        data = dataset.data.to(device)
        _, layer_activations = model(data)

        layer_activations = [ activation.detach() for activation in layer_activations ]

        return layer_activations
    
def save_weights(model, epoch, path="./save/"):
    """
    Save the weights of the model for the given dataset":
        <path>weights_epoch_<epoch>.pt

    Args:
        model: torch.nn.Module, model to evaluate.
        epoch: int, current epoch number.
        path: str, path to the directory where the data is saved.
    """
        
    torch.save(model.state_dict(), path+"weights_epoch_"+str(epoch)+".pt")

def load_weights(model, device, file):
    """
    Load the weights of the model for the given dataset from:
        <path>weights_epoch_<epoch>.pt

    Args:
        model: torch.nn.Module, model to evaluate.
        device: torch.device, device to use for the computation.
        file: str, path to the file containing the weights.
    """

    model.load_state_dict( torch.load(file, map_location=device) )
    model.eval()
