#!/bin/bash
#SBATCH --job-name=contrast_limits
#SBATCH --error=contrast_limits_%j.err
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=compute

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Load the napari-env conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate napari-env

# Run the batch computation
echo "Starting contrast limits computation at $(date)"
echo "Using conda environment: $(conda info --envs | grep '*')"
python compute_contrast_limits_batch.py --csv-file tomo_ids.csv --output-file tomo_ids_with_contrast.csv

echo "Completed contrast limits computation at $(date)" 