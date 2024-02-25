"""
Utility functions for saving and loading data.

save_setup: save the setup to a json file.

load_setup: load the setup from a json file and convert lambda source code to lambda function.
"""
import json
import inspect
import datetime
import os

def save_setup(setup, path, fname=None):
    """
    Save the setup to a json file.

    Args:
        setup: dict, dictionary containing the setup parameters.
        path: str, path to the directory where the data is saved.
        fname: str, name of the file to save the setup.
    """

    if fname is None:
        fname = "setup"

    # Save setup to json
    def default( o ):
        """
        Default function for json.dump to handle non-serializable objects. 
        Specifically we wish to handle entries that are lambda functions. We will save the source code of the function instead.
        """
        # inspect.getsource( value )
        f_str = '='.join( inspect.getsource( o ).split('=')[1:] )[1:]
        return f_str

    with open( path+fname+".json", "w" ) as outfile: 
        json.dump(setup, outfile, default=default, indent=0)
    
def load_setup(folder, path="./save/", fname=None):
    """
    Load the setup from a json file and convert lambda source code to lambda function.

    Args:
        folder: str, subdirectory name where the setup was saved.
        path: str, path to the directory where the data is saved.
    
    Returns:
        loaded_results: dict, dictionary containing the setup parameters and functions.
    """
    if fname is None:
        fname = "setup"

    with open( path+folder+"/"+fname+".json", "r" ) as infile: 
        loaded_results = json.load(infile)
    for key in ["hidden_activation_f", "output_activation_f", "optimizer", "loss_function", "evaluate_correct"]:
        loaded_results[key] = eval( loaded_results[key] )
    
    return loaded_results