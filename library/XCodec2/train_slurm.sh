#!/bin/bash

#SBATCH --job-name=codec
#SBATCH --time=168:00:00 
#SBATCH --nodes=5
#SBATCH --gpus-per-node=8  
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=28  
#SBATCH --mem=1024G 
#SBATCH --exclusive
 


export LOGLEVEL=INFO

export NCCL_DEBUG=INFO

 
export PYTHONWARNINGS="ignore"
srun python train.py log_dir=/path/to/log_dir
