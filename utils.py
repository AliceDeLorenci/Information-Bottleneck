import json
import inspect
import datetime
import os

def temporizer(epoch):
    """
    Determines whether to SKIP saving the activations/weights (or computing MI) for the current epoch.

    Args:
        epoch: int, current epoch number.

    Returns:
        bool, whether to skip the current epoch.
    """

    if epoch < 1000:         # Save for all first 100 epochs # !!!
        return False
    else:
        if not epoch % 10 == 0:
            return True
    # elif epoch < 1000:      # Then every 10 epochs
    #     if not epoch % 10 == 0:
    #         return True
    # else:                   # Then every 100 epochs
    #     if not epoch % 100 == 0:
    #         return True

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
    
def load_setup(path, fname=None):
    """
    Load the setup from a json file and convert lambda source code to lambda function.

    Args:
        folder: str, subdirectory name where the setup was saved.
        path: str, path to the directory where the data is saved.
    
    Returns:
        loaded_results: dict, dictionary containing the setup parameters and functions.
    """
    if fname is None:
        fname = "setup.json"
    elif not fname.endswith(".json"):
        fname = fname+".json"

    with open( path+fname, "r" ) as file: 
        loaded_results = json.load(file)
    for key in ["hidden_activation_f", "output_activation_f", "optimizer", "loss_function", "evaluate_correct"]:
        if key in loaded_results.keys():
            loaded_results[key] = eval( loaded_results[key] )
    
    return loaded_results