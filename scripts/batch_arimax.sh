#!/bin/bash
#SBATCH --job-name=arimax
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=2GB
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/slurm/arimax.log

# Load all modules you need below.
module load pytorch
export PYTHONUSERBASE=/scratch/$ACCOUNT/ed-python-env
export PYTHONPATH=/scratch/$ACCOUNT/ed-python-env/lib/python3.9/site-packages

# Finally run your job.
python scripts/train.py $TARGET $MODEL $FEATURESET $HPO
