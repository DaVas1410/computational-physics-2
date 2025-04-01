#!/bin/bash
#SBATCH --job-name=my_job5
#SBATCH --partition=cpu
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=1G
#SBATCH --output=my_job5.out
#SBATCH --error=my_job5.err

# Your commands go here
source ~/.bashrc
conda activate py310
python test_mp.py
