#!/bin/bash
##SBATCH --job-name=tft
#SBATCH --time=48:00:00
#SBATCH --mem=16GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --output=logs/slurm/tft.log

# Load all modules you need below.
module load pytorch
export PYTHONUSERBASE=/scratch/$ACCOUNT/ed-python-env
export PYTHONPATH=/scratch/$ACCOUNT/ed-python-env/lib/python3.9/site-packages

# Finally run your job.
srun python scripts/train.py $TARGET $MODEL $FEATURESET $HPO
