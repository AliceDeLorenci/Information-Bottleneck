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
    Index smaller than 100 are reserved for synthetic datasets, index larger than 100 are reserved for MNIST datasets.
    """
    setup = dict()
    ######################################## SYNTHETIC ########################################
    if idx < 100:

        if idx == 11:
            setup["comments"] = "original paper setup"
            
            # dataset parameters
            setup["dataset"] = "synthetic"
            setup["train_ratio"] = 0.85

            # network parameters
            setup["hidden_dims"] = [12, 10, 7, 5, 4, 3, 2]                         
            setup["output_dim"] = 1                                         
            
            setup["hidden_activation"] = "tanh"      
            
            setup["output_activation"] = "sigmoid"

            # optimizer
            setup["lr"] = 0.01                             
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = None
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( torch.round(output.reshape(-1)) == target, dtype=torch.float32 )

        elif idx == 21:
            setup["comments"] = "original paper setup [mini-batch][smaller lr]"
            
            # dataset parameters
            setup["dataset"] = "synthetic"
            setup["train_ratio"] = 0.85

            # network parameters
            setup["hidden_dims"] = [12, 10, 7, 5, 4, 3, 2]                         
            setup["output_dim"] = 1                                         
            
            setup["hidden_activation"] = "tanh"      
            
            setup["output_activation"] = "sigmoid"

            # optimizer
            setup["lr"] = 0.001                             
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = 256
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( torch.round(output.reshape(-1)) == target, dtype=torch.float32 )

        elif idx == 31:
            setup["comments"] = "[ibdnn][tanh]"
            
            # dataset parameters
            setup["dataset"] = "synthetic"
            setup["train_ratio"] = 0.85

            # network parameters
            setup["hidden_dims"] = [10, 7, 5, 4, 3]                         
            setup["output_dim"] = 1                                         
            
            setup["hidden_activation"] = "tanh"      
            
            setup["output_activation"] = "sigmoid"

            # optimizer
            setup["lr"] = 0.0004                            
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = 256
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( torch.round(output.reshape(-1)) == target, dtype=torch.float32 )

        elif idx == 41:
            setup["comments"] = "[ibdnn][tanh]"
            
            # dataset parameters
            setup["dataset"] = "synthetic"
            setup["train_ratio"] = 0.85

            # network parameters
            setup["hidden_dims"] = [12, 10, 7, 5, 4, 3]                         
            setup["output_dim"] = 1                                         
            
            setup["hidden_activation"] = "tanh"      
            
            setup["output_activation"] = "sigmoid"

            # optimizer
            setup["lr"] = 0.0004                            
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = 256
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( torch.round(output.reshape(-1)) == target, dtype=torch.float32 )

        elif idx == 42:
            setup["comments"] = "[ibdnn][relu]"
            
            # dataset parameters
            setup["dataset"] = "synthetic"
            setup["train_ratio"] = 0.85

            # network parameters
            setup["hidden_dims"] = [12, 10, 7, 5, 4, 3]                         
            setup["output_dim"] = 1                                         
            
            setup["hidden_activation"] = "relu"      
            
            setup["output_activation"] = "sigmoid"

            # optimizer
            setup["lr"] = 0.0004                            
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = 256
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( torch.round(output.reshape(-1)) == target, dtype=torch.float32 )

        elif idx == 43:
            setup["comments"] = "[ibdnn][relu]"
            
            # dataset parameters
            setup["dataset"] = "synthetic"
            setup["train_ratio"] = 0.85

            # network parameters
            setup["hidden_dims"] = [12, 10, 7, 5, 4, 3]                         
            setup["output_dim"] = 1                                         
            
            setup["hidden_activation"] = "relu"      
            
            setup["output_activation"] = "sigmoid"

            # optimizer
            setup["lr"] = 0.0004                            
            setup["momentum"] = 0.9
            setup["weight_decay"] = 1e-4
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"], weight_decay=setup["weight_decay"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = 256
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( torch.round(output.reshape(-1)) == target, dtype=torch.float32 )

        elif idx == 44:
            setup["comments"] = "[ibdnn][relu]"
            
            # dataset parameters
            setup["dataset"] = "synthetic"
            setup["train_ratio"] = 0.85

            # network parameters
            setup["hidden_dims"] = [12, 10, 7, 5, 4, 3]                         
            setup["output_dim"] = 1                                         
            
            setup["hidden_activation"] = "relu"      
            
            setup["output_activation"] = "sigmoid"

            # optimizer
            setup["lr"] = 0.0004                            
            setup["momentum"] = 0.9
            setup["weight_decay"] = 1e-3
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"], weight_decay=setup["weight_decay"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = 256
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( torch.round(output.reshape(-1)) == target, dtype=torch.float32 )


        elif idx == 12:
            setup["comments"] = "original paper setup but with relu activation function"
            
            # dataset parameters
            setup["dataset"] = "synthetic"
            setup["train_ratio"] = 0.85

            # network parameters
            setup["hidden_dims"] = [12, 10, 7, 5, 4, 3, 2]                         
            setup["output_dim"] = 1                                         
            
            setup["hidden_activation"] = "relu"      
            
            setup["output_activation"] = "sigmoid"

            # optimizer
            setup["lr"] = 0.01                             
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = None
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( torch.round(output.reshape(-1)) == target, dtype=torch.float32 )

        elif idx == 22:
            setup["comments"] = "original paper setup but with relu activation function [mini-batch]"
            
            # dataset parameters
            setup["dataset"] = "synthetic"
            setup["train_ratio"] = 0.85

            # network parameters
            setup["hidden_dims"] = [12, 10, 7, 5, 4, 3, 2]                         
            setup["output_dim"] = 1                                         
            
            setup["hidden_activation"] = "relu"      
            
            setup["output_activation"] = "sigmoid"

            # optimizer
            setup["lr"] = 0.01                            
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = 256
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( torch.round(output.reshape(-1)) == target, dtype=torch.float32 )

        elif idx == 13:
            setup["comments"] = "original paper setup but with weight decay"
            
            # dataset parameters
            setup["dataset"] = "synthetic"
            setup["train_ratio"] = 0.85

            # network parameters
            setup["hidden_dims"] = [12, 10, 7, 5, 4, 3, 2]                         
            setup["output_dim"] = 1                                         
            
            setup["hidden_activation"] = "tanh"      
            
            setup["output_activation"] = "sigmoid"

            # optimizer
            setup["lr"] = 0.01                             
            setup["momentum"] = 0.9
            setup["weight_decay"] = 1e-4
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"], weight_decay=setup["weight_decay"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = None
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( torch.round(output.reshape(-1)) == target, dtype=torch.float32 )

        elif idx == 23:
            setup["comments"] = "original paper setup but with weight decay [mini-batch][smaller lr]"
            
            # dataset parameters
            setup["dataset"] = "synthetic"
            setup["train_ratio"] = 0.85

            # network parameters
            setup["hidden_dims"] = [12, 10, 7, 5, 4, 3, 2]                         
            setup["output_dim"] = 1                                         
            
            setup["hidden_activation"] = "tanh"      
            
            setup["output_activation"] = "sigmoid"

            # optimizer
            setup["lr"] = 0.001                             
            setup["momentum"] = 0.9
            setup["weight_decay"] = 1e-4
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"], weight_decay=setup["weight_decay"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = 256
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( torch.round(output.reshape(-1)) == target, dtype=torch.float32 )

        elif idx == 14:
            setup["comments"] = "original paper setup but with weight decay"
            
            # dataset parameters
            setup["dataset"] = "synthetic"
            setup["train_ratio"] = 0.85

            # network parameters
            setup["hidden_dims"] = [12, 10, 7, 5, 4, 3, 2]                         
            setup["output_dim"] = 1                                         
            
            setup["hidden_activation"] = "tanh"      
            
            setup["output_activation"] = "sigmoid"

            # optimizer
            setup["lr"] = 0.01                             
            setup["momentum"] = 0.9
            setup["weight_decay"] = 1e-3
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"], weight_decay=setup["weight_decay"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = None
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( torch.round(output.reshape(-1)) == target, dtype=torch.float32 )

        elif idx == 24:
            setup["comments"] = "original paper setup but with weight decay [mini-batch][smaller lr]"
            
            # dataset parameters
            setup["dataset"] = "synthetic"
            setup["train_ratio"] = 0.85

            # network parameters
            setup["hidden_dims"] = [12, 10, 7, 5, 4, 3, 2]                         
            setup["output_dim"] = 1                                         
            
            setup["hidden_activation"] = "tanh"      
            
            setup["output_activation"] = "sigmoid"

            # optimizer
            setup["lr"] = 0.001                             
            setup["momentum"] = 0.9
            setup["weight_decay"] = 1e-3
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"], weight_decay=setup["weight_decay"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = 256
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( torch.round(output.reshape(-1)) == target, dtype=torch.float32 )

        elif idx == 1:
            # - Synthetic dataset
            # - tanh activation function
            # - Test accuracy: 0.9634 in 10000 epochs

            # dataset parameters
            setup["dataset"] = "synthetic"
            setup["train_ratio"] = 0.8

            # network parameters
            setup["hidden_dims"] = [10, 7, 5, 4, 3]                         
            setup["output_dim"] = 1                                         
            
            setup["hidden_activation"] = "tanh"      
            
            setup["output_activation"] = "sigmoid"

            # optimizer
            setup["lr"] = 0.01                             
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = None
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
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
            
            setup["hidden_activation"] = "relu"
            
            setup["output_activation"] = "sigmoid"

            # optimizer
            setup["lr"] = 0.01                             
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = None
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( torch.round(output.reshape(-1)) == target, dtype=torch.float32 )

        elif idx == 20:
            # - Synthetic dataset
            # - relu activation function
            # - Test accuracy:  0.9780 in 10000 epochs (activations-20240225-161551)

            # dataset parameters
            setup["dataset"] = "synthetic"
            setup["train_ratio"] = 0.8

            # network parameters
            setup["hidden_dims"] = [10, 7, 5, 4, 3]                         
            setup["output_dim"] = 1                                         
            
            setup["hidden_activation"] = "relu"
            
            setup["output_activation"] = "sigmoid"

            # optimizer
            setup["lr"] = 0.1                         
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = None
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
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
            
            setup["hidden_activation"] = "tanh" 
            
            setup["output_activation"] = "sigmoid"   

            # optimizer
            setup["lr"] = 0.01                             
            setup["momentum"] = 0.9
            setup["weight_decay"] = 0.0001
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"], weight_decay=setup["weight_decay"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = None
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
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
                
                setup["hidden_activation"] = "tanh"   
                
                setup["output_activation"] = "sigmoid"  

                # optimizer
                setup["lr"] = 0.01                             
                setup["momentum"] = 0.9
                setup["weight_decay"] = 0.001
                setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"], weight_decay=setup["weight_decay"] )

                # training configuration
                setup["n_epochs"] = 10000
                setup["batch_size"] = None
                setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
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
            
            setup["hidden_activation"] = "tanh"
            
            setup["output_activation"] = "log_softmax"

            # optimizer
            setup["lr"] = 0.0004   
            setup["optimizer"] = lambda parameters: torch.optim.Adam( parameters, lr=setup["lr"] )

            # training configuration
            setup["n_epochs"] = 5000
            setup["batch_size"] = 256
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.nll_loss(output, target, reduction=reduction)
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
            
            setup["hidden_activation"] = "tanh"
            
            setup["output_activation"] = "sigmoid"

            # optimizer
            setup["lr"] = 0.01                             
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = None
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
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
            
            setup["hidden_activation"] = "tanh" 
            
            setup["output_activation"] = "sigmoid"

            # optimizer
            setup["lr"] = 1e-2                             
            setup["momentum"] = 0.9
            setup["weight_decay"] = 1e-3
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"], weight_decay=setup["weight_decay"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = None
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.binary_cross_entropy(output.reshape(-1), target.float(), reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( torch.round(output.reshape(-1)) == target, dtype=torch.float32 )
        
        else:
            raise ValueError("Invalid index.")
    
    ######################################## MNIST ########################################
    elif idx >= 100 and idx<1000:  
        
        if idx == 101:
            # - MNIST
            # - Test accuracy: 
            
            # dataset parameters
            setup["dataset"] = "mnist"
            setup["train_ratio"] = 0.8

            # network parameters
            setup["hidden_dims"] = [512, 256, 64, 64, 32]                                   # hidden layers sizes
            setup["output_dim"] = 10                                                        # output layer size
            
            setup["hidden_activation"] = "tanh"
            
            setup["output_activation"] = "log_softmax"

            # optimizer
            setup["lr"] = 0.01  
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 1000
            setup["batch_size"] = 256
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.nll_loss(output, target, reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( output.argmax(dim=1) == target, dtype=torch.float32 )

        elif idx == 111:
            # - MNIST
            # - Test accuracy: 
            
            # dataset parameters
            setup["dataset"] = "mnist"
            setup["train_ratio"] = 0.8

            # network parameters
            setup["hidden_dims"] = [512, 256, 64, 64, 32]                                   # hidden layers sizes
            setup["output_dim"] = 10                                                        # output layer size
            
            setup["hidden_activation"] = "tanh"
            
            setup["output_activation"] = "log_softmax"

            # optimizer
            setup["lr"] = 0.01  
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 1000
            setup["batch_size"] = None
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.nll_loss(output, target, reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( output.argmax(dim=1) == target, dtype=torch.float32 )
        
        elif idx == 102:
            # - MNIST
            # - Test accuracy: 
            
            # dataset parameters
            setup["dataset"] = "mnist"
            setup["train_ratio"] = 6/7

            # network parameters
            setup["hidden_dims"] = [1024, 20, 20, 20]                                       # hidden layers sizes
            setup["output_dim"] = 10                                                        # output layer size
            
            setup["hidden_activation"] = "tanh"
            
            setup["output_activation"] = "log_softmax"

            # optimizer
            setup["lr"] = 0.001  
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = 128
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.nll_loss(output, target, reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( output.argmax(dim=1) == target, dtype=torch.float32 )

        elif idx == 112:
                # - MNIST
                # - Test accuracy: 
                
                # dataset parameters
                setup["dataset"] = "mnist"
                setup["train_ratio"] = 6/7

                # network parameters
                setup["hidden_dims"] = [1024, 20, 20, 20]                                       # hidden layers sizes
                setup["output_dim"] = 10                                                        # output layer size
                
                setup["hidden_activation"] = "tanh"
                
                setup["output_activation"] = "log_softmax"

                # optimizer
                setup["lr"] = 0.001  
                setup["momentum"] = 0.9
                setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

                # training configuration
                setup["n_epochs"] = 10000
                setup["batch_size"] = None
                setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.nll_loss(output, target, reduction=reduction)
                setup["evaluate_correct"] = lambda output, target: torch.sum( output.argmax(dim=1) == target, dtype=torch.float32 )
        
        elif idx == 103:
            # - MNIST
            # - Test accuracy: 
            
            # dataset parameters
            setup["dataset"] = "mnist"
            setup["train_ratio"] = 6/7

            # network parameters
            setup["hidden_dims"] = [1024, 20, 20, 20]                                       # hidden layers sizes
            setup["output_dim"] = 10                                                        # output layer size
            
            setup["hidden_activation"] = "relu"
            
            setup["output_activation"] = "log_softmax"

            # optimizer
            setup["lr"] = 0.001  
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = 128
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.nll_loss(output, target, reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( output.argmax(dim=1) == target, dtype=torch.float32 )

        elif idx == 113:
            # - MNIST
            # - Test accuracy: 
            
            # dataset parameters
            setup["dataset"] = "mnist"
            setup["train_ratio"] = 6/7

            # network parameters
            setup["hidden_dims"] = [1024, 20, 20, 20]                                       # hidden layers sizes
            setup["output_dim"] = 10                                                        # output layer size
            
            setup["hidden_activation"] = "relu"
            
            setup["output_activation"] = "log_softmax"

            # optimizer
            setup["lr"] = 0.001  
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = None
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.nll_loss(output, target, reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( output.argmax(dim=1) == target, dtype=torch.float32 )
        
        elif idx == 123:
            # - MNIST
            # - Test accuracy: 
            
            # dataset parameters
            setup["dataset"] = "mnist"
            setup["train_ratio"] = 6/7

            # network parameters
            setup["hidden_dims"] = [1024, 20, 20, 20]                                       # hidden layers sizes
            setup["output_dim"] = 10                                                        # output layer size
            
            setup["hidden_activation"] = "relu"
            
            setup["output_activation"] = "log_softmax"

            # optimizer
            setup["lr"] = 0.0005  
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = 128
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.nll_loss(output, target, reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( output.argmax(dim=1) == target, dtype=torch.float32 )

        elif idx == 133:
            # - MNIST
            # - Test accuracy: 
            
            # dataset parameters
            setup["dataset"] = "mnist"
            setup["train_ratio"] = 6/7

            # network parameters
            setup["hidden_dims"] = [1024, 20, 20, 20]                                       # hidden layers sizes
            setup["output_dim"] = 10                                                        # output layer size
            
            setup["hidden_activation"] = "relu"
            
            setup["output_activation"] = "log_softmax"

            # optimizer
            setup["lr"] = 0.001  
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = 1024
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.nll_loss(output, target, reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( output.argmax(dim=1) == target, dtype=torch.float32 )

        else:
            raise ValueError("Invalid index.")
    
    ######################################## MNIST ########################################
    elif idx >= 1000:

        if idx == 1001:
            # - MNIST
            # - Test accuracy: 

            setup["comments"] = "[tanh][full batch]"
            
            # dataset parameters
            setup["dataset"] = "mnist"
            setup["train_ratio"] = 0.85

            # network parameters
            setup["hidden_dims"] = [256, 16, 16, 16]                                  
            setup["output_dim"] = 10                                                        
            
            setup["hidden_activation"] = "tanh"
            
            setup["output_activation"] = "log_softmax"

            # optimizer
            setup["lr"] = 0.001  
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = None
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.nll_loss(output, target, reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( output.argmax(dim=1) == target, dtype=torch.float32 )

        if idx == 1011:
            # - MNIST
            # - Test accuracy: 

            setup["comments"] = "[tanh][mini batch]"
            
            # dataset parameters
            setup["dataset"] = "mnist"
            setup["train_ratio"] = 0.85

            # network parameters
            setup["hidden_dims"] = [256, 16, 16, 16]                                  
            setup["output_dim"] = 10                                                        
            
            setup["hidden_activation"] = "tanh"
            
            setup["output_activation"] = "log_softmax"

            # optimizer
            setup["lr"] = 0.01  
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = 128
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.nll_loss(output, target, reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( output.argmax(dim=1) == target, dtype=torch.float32 )

        if idx == 1021:
            # - MNIST
            # - Test accuracy: 

            setup["comments"] = "[tanh][mini batch][epoch is iteration]"
            
            # dataset parameters
            setup["dataset"] = "mnist"
            setup["train_ratio"] = 0.85

            # network parameters
            setup["hidden_dims"] = [256, 16, 16, 16]                                  
            setup["output_dim"] = 10                                                        
            
            setup["hidden_activation"] = "tanh"
            
            setup["output_activation"] = "log_softmax"

            # optimizer
            setup["lr"] = 0.001  
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = 1024
            setup["epoch_is_iteration"] = True  # one epoch equals one pass through the training set (False) or one epoch equals one iteration/mini-batch (True)
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.nll_loss(output, target, reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( output.argmax(dim=1) == target, dtype=torch.float32 )

        if idx == 1101:
            # - MNIST
            # - Test accuracy: 

            setup["comments"] = "[ibsgd][tanh]"
            
            # dataset parameters
            setup["dataset"] = "mnist"
            setup["train_ratio"] = 0.85

            # network parameters
            setup["hidden_dims"] = [1024, 20, 20, 20]                                  
            setup["output_dim"] = 10                                                        
            
            setup["hidden_activation"] = "tanh"
            
            setup["output_activation"] = "log_softmax"

            # optimizer
            setup["lr"] = 0.001  
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = 128
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.nll_loss(output, target, reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( output.argmax(dim=1) == target, dtype=torch.float32 )


        if idx == 1002:
            # - MNIST
            # - Test accuracy: 

            setup["comments"] = "[relu][full batch]"
            
            # dataset parameters
            setup["dataset"] = "mnist"
            setup["train_ratio"] = 0.85

            # network parameters
            setup["hidden_dims"] = [256, 16, 16, 16]                                  
            setup["output_dim"] = 10                                                        
            
            setup["hidden_activation"] = "relu"
            
            setup["output_activation"] = "log_softmax"

            # optimizer
            setup["lr"] = 0.001  
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = None
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.nll_loss(output, target, reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( output.argmax(dim=1) == target, dtype=torch.float32 )

        if idx == 1012:
            # - MNIST
            # - Test accuracy: 

            setup["comments"] = "[relu][mini batch]"
            
            # dataset parameters
            setup["dataset"] = "mnist"
            setup["train_ratio"] = 0.85

            # network parameters
            setup["hidden_dims"] = [256, 16, 16, 16]                                  
            setup["output_dim"] = 10                                                        
            
            setup["hidden_activation"] = "relu"
            
            setup["output_activation"] = "log_softmax"

            # optimizer
            setup["lr"] = 0.01  
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = 128
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.nll_loss(output, target, reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( output.argmax(dim=1) == target, dtype=torch.float32 )

        if idx == 1022:
            # - MNIST
            # - Test accuracy: 

            setup["comments"] = "[tanh][mini batch][epoch is iteration]"
            
            # dataset parameters
            setup["dataset"] = "mnist"
            setup["train_ratio"] = 0.85

            # network parameters
            setup["hidden_dims"] = [256, 16, 16, 16]                                  
            setup["output_dim"] = 10                                                        
            
            setup["hidden_activation"] = "relu"
            
            setup["output_activation"] = "log_softmax"

            # optimizer
            setup["lr"] = 0.001  
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = 1024
            setup["epoch_is_iteration"] = True  # one epoch equals one pass through the training set (False) or one epoch equals one iteration/mini-batch (True)
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.nll_loss(output, target, reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( output.argmax(dim=1) == target, dtype=torch.float32 )

        if idx == 1102:
            # - MNIST
            # - Test accuracy: 

            setup["comments"] = "[ibsgd][relu]"
            
            # dataset parameters
            setup["dataset"] = "mnist"
            setup["train_ratio"] = 0.85

            # network parameters
            setup["hidden_dims"] = [1024, 20, 20, 20]                                  
            setup["output_dim"] = 10                                                        
            
            setup["hidden_activation"] = "relu"
            
            setup["output_activation"] = "log_softmax"

            # optimizer
            setup["lr"] = 0.001  
            setup["momentum"] = 0.9
            setup["optimizer"] = lambda parameters: torch.optim.SGD( parameters, lr=setup["lr"], momentum=setup["momentum"] )

            # training configuration
            setup["n_epochs"] = 10000
            setup["batch_size"] = 128
            setup["loss_function"] = lambda output, target, reduction='mean': torch.nn.functional.nll_loss(output, target, reduction=reduction)
            setup["evaluate_correct"] = lambda output, target: torch.sum( output.argmax(dim=1) == target, dtype=torch.float32 )

        
    else:
        raise ValueError("Invalid index.")
    
    return setup


