#!/bin/bash
#SBATCH --job-name=baseline_raw_14b
#SBATCH -o output/run_%j.out
#SBATCH -e output/run_%j.err
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint="vram40"

module load miniconda/22.11.1-1
conda activate eqa-baseline

python raw_vlm_exp.py -cf cfg/raw_test_13b.yaml