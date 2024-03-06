# Bash script to run in parallel multiple repetitions of PIPELINE.py with the same setup

# Usage: 
# bash train.sh <start_idx> <n_repetitions> 

# <start_idx> : index of the first repetition
# <n_repetitions> : number of repetitions

SETUP=$((1102))

START=$1
END=$(( $1 + $2 ))

for ((i=$START; i<$END; i++))
do  
    # python3 PIPELINE.py --shuffle --verbose=0 --subdir=$i --setup_idx=$SETUP --warmup --threshold=0.65& # synthetic
    python3 PIPELINE.py --shuffle --verbose=0 --subdir=$i --setup_idx=$SETUP --data=test --noise_variance=0.1 --temporize&
done 
wait