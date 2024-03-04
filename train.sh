# Bash script to run in parallel multiple repetitions of TRAIN.py with the same setup

# Usage: 
# bash train.sh <start_idx> <n_repetitions> 

# <start_idx> : index of the first repetition
# <n_repetitions> : number of repetitions

SETUP=$((1112))

START=$1
END=$(( $1 + $2 ))

for ((i=$START; i<$END; i++))
do  
    # python3 TRAIN.py --shuffle --verbose=0 --subdir=$i --setup_idx=$SETUP --warmup & # sythetic
    python3 TRAIN.py --shuffle --verbose=0 --subdir=$i --setup_idx=$SETUP & # mnist
done
wait
for ((i=$START; i<$END; i++))
do  
    # python3 MI.py --subdir=$i --setup_idx=14 & # synthetic
    python3 MI.py --subdir=$i --setup_idx=$SETUP --data=test --noise_variance=0.1 & # mnist
done
wait