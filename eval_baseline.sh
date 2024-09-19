#!/bin/bash
#SBATCH --job-name=eval_new_13b
#SBATCH -o output/run_eval_new_13b_%j.out
#SBATCH -e output/run_eval_new_13b_%j.err
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint="vram40"

module load miniconda/22.11.1-1
conda activate eqa-baseline

python run_vlm_exp.py -cf cfg/eval_baseline_13b.yaml