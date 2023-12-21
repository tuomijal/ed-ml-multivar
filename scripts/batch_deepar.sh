#!/bin/bash
#SBATCH --job-name=deepar
#SBATCH --time=24:00:00
#SBATCH --mem=4GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=logs/slurm/deepar.log

# Load all modules you need below.
module load pytorch
export PYTHONUSERBASE=/scratch/$ACCOUNT/ed-python-env
export PYTHONPATH=/scratch/$ACCOUNT/ed-python-env/lib/python3.9/site-packages

# As suggested here: https://stackoverflow.com/questions/49875588/importerror-lib64-libstdc-so-6-version-cxxabi-1-3-9-not-found
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/CSC_CONTAINER/miniconda/envs/env1/lib

# Finally run your job.
python scripts/train.py $TARGET $MODEL $FEATURESET $HPO
