# Bash script to run in parallel multiple repetitions of a same setup

# Usage: 
# bash compute_activations.sh <n_repetitions> <setup_idx> <verbose> <directory>

# <n_repetitions> : number of repetitions
# <setup_idx> : index of the setup according to setups.py
# <verbose> : verbosity level (default: 0)
# <directory> : directory where the results will be saved (default: setup-<setup_idx>)

START=1
END=$1
for ((i=$START; i<=$END; i++))
do  
    python3 compute_activations.py $2 $3 $4 &
done
wait