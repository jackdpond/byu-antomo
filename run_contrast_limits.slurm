#!/bin/bash
#SBATCH --job-name=contrast_limits
#SBATCH --error=contrast_limits_%j.err
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=msg
#SBATCH --v100:1

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Load any necessary modules (adjust for your system)
# module load cuda/11.8
# module load gcc/9.3.0

# Activate conda environment
source ~/.bashrc
conda activate napari-env

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run the batch computation
echo "Starting contrast limits computation at $(date)"
echo "Using conda environment: $(conda info --envs | grep '*')"
python compute_contrast_limits_batch.py --csv-file tomo_ids.csv

echo "Completed contrast limits computation at $(date)" 