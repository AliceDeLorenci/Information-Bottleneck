"""
Contains the setup dictionaries for the different experiments.

setup_lookup: returns the desired setup dictionary.
"""
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
        # - tanh activation function
        # - Test accuracy: 0.9634 in 10000 epochs

        # dataset parameters
        setup["dataset"] = "synthetic"
        setup["train_ratio"] = 0.8

        # network parameters
        setup["hidden_dims"] = [10, 7, 5, 4, 3]                         
        setup["output_dim"] = 1                                         
        setup["hidden_activation_f"] = lambda input: F.tanh(input)
        setup["hidden_activation"] = "tanh"      
        setup["output_activation_f"] = lambda input: F.sigmoid(input)   
        setup["output_activation"] = "sigmoid"

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
        # - relu activation function
        # - Test accuracy:  0.9780 in 10000 epochs (activations-20240225-161551)

        # dataset parameters
        setup["dataset"] = "synthetic"
        setup["train_ratio"] = 0.8

        # network parameters
        setup["hidden_dims"] = [10, 7, 5, 4, 3]                         
        setup["output_dim"] = 1                                         
        setup["hidden_activation_f"] = lambda input: F.relu(input)     
        setup["hidden_activation"] = "relu"
        setup["output_activation_f"] = lambda input: F.sigmoid(input)   
        setup["output_activation"] = "sigmoid"

        # optimizer
        setup["lr"] = 0.01                             
        setup["momentum"] = 0.9
        setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

        # training configuration
        setup["n_epochs"] = 10000
        setup["batch_size"] = None
        setup["loss_function"] = lambda output, target, reduction='mean': F.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
        setup["evaluate_correct"] = lambda output, target: torch.sum( torch.round(output.reshape(-1)) == target, dtype=torch.float32 )
    
    elif idx == 3:
        # - Synthetic dataset
        # - tanh activation function with weight decay
        # - Test accuracy: 0.9634 in 10000 epochs

        # dataset parameters
        setup["dataset"] = "synthetic"
        setup["train_ratio"] = 0.8

        # network parameters
        setup["hidden_dims"] = [10, 7, 5, 4, 3]                         
        setup["output_dim"] = 1                                         
        setup["hidden_activation_f"] = lambda input: F.tanh(input)    
        setup["hidden_activation"] = "tanh" 
        setup["output_activation_f"] = lambda input: F.sigmoid(input)
        setup["output_activation"] = "sigmoid"   

        # optimizer
        setup["lr"] = 0.01                             
        setup["momentum"] = 0.9
        setup["weight_decay"] = 0.0001
        setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"], weight_decay=setup["weight_decay"] )

        # training configuration
        setup["n_epochs"] = 10000
        setup["batch_size"] = None
        setup["loss_function"] = lambda output, target, reduction='mean': F.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
        setup["evaluate_correct"] = lambda output, target: torch.sum( torch.round(output.reshape(-1)) == target, dtype=torch.float32 )

    elif idx == 4:
            # - Synthetic dataset
            # - tanh activation function with weight decay (larger)
            # - Test accuracy: 0.9707 in 10000 epochs

            # dataset parameters
            setup["dataset"] = "synthetic"
            setup["train_ratio"] = 0.8

            # network parameters
            setup["hidden_dims"] = [10, 7, 5, 4, 3]                         
            setup["output_dim"] = 1                                         
            setup["hidden_activation_f"] = lambda input: F.tanh(input)   
            setup["hidden_activation"] = "tanh"   
            setup["output_activation_f"] = lambda input: F.sigmoid(input) 
            setup["output_activation"] = "sigmoid"  

            # optimizer
            setup["lr"] = 0.01                             
            setup["momentum"] = 0.9
            setup["weight_decay"] = 0.001
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"], weight_decay=setup["weight_decay"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = None
            setup["loss_function"] = lambda output, target, reduction='mean': F.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( torch.round(output.reshape(-1)) == target, dtype=torch.float32 )

    elif idx == None:
        # - Synthetic dataset with 2 classes
        # - tanh activation function
        # - Test accuracy: 

        # dataset parameters
        setup["dataset"] = "synthetic"
        setup["train_ratio"] = 0.85

        # network parameters
        setup["hidden_dims"] = [10, 7, 5, 4, 3]                         
        setup["output_dim"] = 2                                       
        setup["hidden_activation_f"] = lambda input: F.tanh(input)      
        setup["hidden_activation"] = "tanh"
        setup["output_activation_f"] = lambda input: F.log_softmax(input, dim=1)   
        setup["output_activation"] = "log_softmax"

        # optimizer
        setup["lr"] = 0.0004   
        setup["optimizer"] = lambda parameters: torch.optim.Adam( parameters, lr=setup["lr"] )

        # training configuration
        setup["n_epochs"] = 5000
        setup["batch_size"] = 256
        setup["loss_function"] = lambda output, target, reduction='mean': F.nll_loss(output, target, reduction=reduction)
        setup["evaluate_correct"] = lambda output, target: torch.sum( output.argmax(dim=1) == target, dtype=torch.float32 )

    elif idx == None:
        # - Synthetic dataset
        # - tanh activation function
        # - Test accuracy: 0.9426

        # dataset parameters
        setup["dataset"] = "synthetic"
        setup["train_ratio"] = 0.8

        # network parameters
        setup["hidden_dims"] = [10, 7, 5, 4, 3]                         
        setup["output_dim"] = 1                                         
        setup["hidden_activation_f"] = lambda input: F.tanh(input)      
        setup["hidden_activation"] = "tanh"
        setup["output_activation_f"] = lambda input: F.sigmoid(input)   
        setup["output_activation"] = "sigmoid"

        # optimizer
        setup["lr"] = 0.01                             
        setup["momentum"] = 0.9
        setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

        # training configuration
        setup["n_epochs"] = 10000
        setup["batch_size"] = None
        setup["loss_function"] = lambda output, target, reduction='mean': F.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
        setup["evaluate_correct"] = lambda output, target: torch.sum( torch.round(output.reshape(-1)) == target, dtype=torch.float32 )
    
    elif idx == None:
        # - Synthetic dataset
        # - tanh activation function + weight decay
        # - Test accuracy: 

        # dataset parameters
        setup["dataset"] = "synthetic"
        setup["train_ratio"] = 0.8

        # network parameters
        setup["hidden_dims"] = [10, 7, 5, 4, 3]                         
        setup["output_dim"] = 1                                         
        setup["hidden_activation_f"] = lambda input: F.tanh(input)     
        setup["hidden_activation"] = "tanh" 
        setup["output_activation_f"] = lambda input: F.sigmoid(input)   
        setup["output_activation"] = "sigmoid"

        # optimizer
        setup["lr"] = 1e-2                             
        setup["momentum"] = 0.9
        setup["weight_decay"] = 1e-3
        setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"], weight_decay=setup["weight_decay"] )

        # training configuration
        setup["n_epochs"] = 10000
        setup["batch_size"] = None
        setup["loss_function"] = lambda output, target, reduction='mean': F.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
        setup["evaluate_correct"] = lambda output, target: torch.sum( torch.round(output.reshape(-1)) == target, dtype=torch.float32 )
    
    else:
        raise ValueError("Invalid index.")
    
    return setup


