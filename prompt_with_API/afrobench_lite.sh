#!/bin/bash

#SBATCH --partition=main-cpu      # Adjust partition as needed
#SBATCH --cpus-per-task=8           # Request 6 CPUs
#SBATCH --mem=64G                   # Request 32GB RAM
#SBATCH --time=32:00:00             # Set max runtime
#SBATCH -o /network/scratch/j/jessica.ojo/slurm-%j.out  # Log output

# Load necessary modules
module load miniconda/3

# Initialize Conda
source /home/mila/j/jessica.ojo/scratch/miniconda3/etc/profile.d/conda.sh

# Activate the environment
conda activate custom
export GEMINI_API_KEY="AIzaSyDQh9RyL-0JGUy6Fh7Rr_uuQL0UW-4orLc"
export HF_HOME='~/scratch/.cache/huggingface'
export HF_DATASETS_CACHE='~/scratch/.cache/huggingface'
export TOKENIZERS_PARALLELISM=false
#export CUDA_VISIBLE_DEVICES=0,1,2,3

# run script
python run.py --tasks afrobench_lite --model 'gemini-2.0-flash' --output './gemini_lite'
