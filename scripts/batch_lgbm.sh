#!/bin/bash
#SBATCH --job-name=lgbm
#SBATCH --time=36:00:00
#SBATCH --mem=150GB
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --output=logs/slurm/lgbm.log

# Load all modules you need below.
module load pytorch
export PYTHONPATH=/scratch/$ACCOUNT/ed-python-env/python3.9/site-packages

# Finally run your job.
python scripts/train.py $TARGET $MODEL $FEATURESET $HPO
