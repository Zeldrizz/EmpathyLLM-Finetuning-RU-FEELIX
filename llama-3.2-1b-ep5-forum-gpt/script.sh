#!/bin/bash
#SBATCH --job-name=llama-3.1-8b-RU-finetune-STCD
#SBATCH --error=logs/data.err
#SBATCH --output=logs/data.log
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8

module load Python/Anaconda_v03.2023
module load CUDA/12.2
source activate my_py_env1

python main.py