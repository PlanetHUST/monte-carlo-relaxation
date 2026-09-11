#!/bin/bash
#SBATCH --job-name=annihilation
#SBATCH --output=out.txt
#SBATCH --error=err.txt
#SBATCH --partition=cpu-pgmf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=50

python -u scripts/run_annihilation.py
