#!/bin/bash
#SBATCH --job-name=llama-3.1-8b-RU-finetune-STCD-GMPM-PSYCH-ANS
#SBATCH --error=logs/data.err
#SBATCH --output=logs/data.log
#SBATCH --time=60:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4

module load Python/Anaconda_v03.2023
module load CUDA/12.2
source activate my_py_env1

torchrun --nproc_per_node=2 main.py