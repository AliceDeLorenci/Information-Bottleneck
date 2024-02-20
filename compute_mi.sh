from dataset import *
from mi import * 
from nn import *

if __name__ == '__main__':
    # Load the dataset
    dataset = buildDatasets( *loadSyntheticData(file="data/synthetic/var_u.mat"), name="synthetic" )
    
    # Compute the mutual information
    mi = compute_mi(dataset)
    # Print the mutual information
    print(mi)
```