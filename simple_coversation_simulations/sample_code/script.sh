#!/bin/bash
#SBATCH --job-name=simulate_model
#SBATCH --error=logs/simulate_model.err
#SBATCH --output=logs/simulate_model.log
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

module load Python/Anaconda_v03.2023
module load CUDA/12.2
source activate my_py_env1

python simulate.py