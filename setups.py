import torch
import torch.nn as nn
import torch.nn.functional as F

def setup_lookup(idx):
    """
    Return the desired setup dictionary.
    """
    setup = dict()
    if idx == 1:
        # - Synthetic dataset
        # - Test accuracy: 0.9426

        # dataset parameters
        setup["dataset"] = "synthetic"
        setup["train_ratio"] = 0.8

        # network parameters
        setup["hidden_dims"] = [10, 7, 5, 4, 3]                         
        setup["output_dim"] = 1                                         
        setup["hidden_activation_f"] = lambda input: F.tanh(input)      
        setup["output_activation_f"] = lambda input: F.sigmoid(input)   

        # optimizer
        setup["lr"] = 0.01                             
        setup["momentum"] = 0.9
        setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

        # training configuration
        setup["n_epochs"] = 10000
        setup["batch_size"] = None
        setup["loss_function"] = lambda output, target, reduction='mean': F.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
        setup["evaluate_correct"] = lambda output, target: torch.sum( torch.round(output.reshape(-1)) == target, dtype=torch.float32 )

    elif idx == 2:
        # - Synthetic dataset
        # - Test accuracy: 0.9635

        # dataset parameters
        setup["dataset"] = "synthetic"
        setup["train_ratio"] = 0.8

        # network parameters
        setup["hidden_dims"] = [10, 7, 5, 4, 3]                         
        setup["output_dim"] = 1                                         
        setup["hidden_activation_f"] = lambda input: F.relu(input)      
        setup["output_activation_f"] = lambda input: F.sigmoid(input)   

        # optimizer
        setup["lr"] = 0.01                             
        setup["momentum"] = 0.9
        setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

        # training configuration
        setup["n_epochs"] = 10000
        setup["batch_size"] = None
        setup["loss_function"] = lambda output, target, reduction='mean': F.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
        setup["evaluate_correct"] = lambda output, target: torch.sum( torch.round(output.reshape(-1)) == target, dtype=torch.float32 )
    
    return setup


