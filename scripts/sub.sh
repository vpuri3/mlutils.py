#!/bin/bash

#SBATCH -N 1
#SBATCH -t 16:00:00
#SBATCH -p GPU
#SBATCH --gpus=v100-32:8
#SBATCH --output=slurm-%j.out

set -x
source $HOME/.bash_profile
cd /path/to/mlutils.py
module load cuda anaconda3
conda activate GeomLearning

EXP_NAME="exp"
torchrun \
    --nproc-per-node gpu \
    -m project \
    --exp_name ${EXP_NAME} \
    --train true \
    --epochs 100

exit
#