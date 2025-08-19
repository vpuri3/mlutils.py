#!/bin/bash

#SBATCH -N 2
#SBATCH -t 8:00:00
#SBATCH -p GPU
#SBATCH --gpus=v100-32:16

set -x
source $HOME/.bash_profile
cd /path/to/mlutils.py
module load cuda anaconda3
conda activate GeomLearning

MASTER_ADDR=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)
NNODES=${SLURM_NNODES}
NPROC_PER_NODE=${SLURM_GPUS_ON_NODE}
MASTER_PORT=29500

echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Number of Nodes: $NNODES"
echo "GPUs per Node: $NPROC_PER_NODE"

EXP_NAME="exp"
torchrun \
    --nproc-per-node gpu \
    --nnodes=$NNODES \
    --rdzv-id=$SLURM_JOB_ID \
    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m project \
    --exp_name $EXP_NAME \
    --train true \
    --epochs 100

exit
#