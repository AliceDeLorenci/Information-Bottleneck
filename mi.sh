# Bash script to run in parallel multiple repetitions of MI.py with the same setup

# Usage: 
# bash main.sh <start_idx> <n_repetitions> 

# <start_idx> : index of the first repetition
# <n_repetitions> : number of repetitions

SETUP=$((1011))

START=$1
END=$(( $1 + $2 ))

for ((i=$START; i<$END; i++))
do  
    # python3 MI.py --subdir=$i --setup_idx=$SETUP & # synthetic
    python3 MI.py --subdir=$i --setup_idx=$SETUP --data=test --noise_variance=0.1 & # mnist
done
wait
