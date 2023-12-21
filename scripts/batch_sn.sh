#!/bin/bash
##SBATCH --job-name=snaive
#SBATCH --time=00:15:00
#SBATCH --mem=2GB
#SBATCH --partition=small
##SBATCH --output=logs/slurm/snaive.log

# Load all modules you need below.
module load pytorch
export PYTHONUSERBASE=/scratch/$ACCOUNT/ed-python-env 
export PYTHONPATH=/scratch/$ACCOUNT/ed-python-env/lib/python3.9/site-packages

# Finally run your job.
python scripts/train.py $TARGET $MODEL $FEATURESET $HPO
