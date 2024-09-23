#!/bin/bash
#SBATCH --job-name=baseline_vlm_7b
#SBATCH -o output/vlm_%j.out
#SBATCH -e output/vlm_%j.err
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint="vram40"

module load miniconda/22.11.1-1
module load gcc/13.2.0
conda activate /work/pi_chuangg_umass_edu/yuncong/conda_envs/eqa-baseline

python run_vlm_exp.py -cf cfg/eval_baseline_vlm.yaml