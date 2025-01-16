#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --error=logs/test_model.err
#SBATCH --output=logs/test_model.log
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2

module load Python/Anaconda_v03.2023
module load CUDA/12.2
source activate my_py_env1

python run_model.py