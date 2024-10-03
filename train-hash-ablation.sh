#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=128gb
#SBATCH --qos=high
#SBATCH --output=test-hash.o

# ablation parameters
shareds=("250 500 1000")
lsrs=("1.2 1.5 2 3 4") #length to shared ratios
dim=256

mkdir -p ./runs/

for shared in $shareds
do
    for lsr in $lsrs
    do
        len=`echo "$shared * $lsr" | bc` # need to use bc for float math
        len=`printf "%.0f" $len` # convert to int
        python train-hash.py -e chunk -d $dim -l $len -s $shared > ./runs/${shared}-${lsr}.out
    done
done