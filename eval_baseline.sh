#!/bin/bash
#SBATCH --job-name=eval_new_gpt
#SBATCH -o output/run_eval_new_gpt_%j.out
#SBATCH -e output/run_eval_new_gpt_%j.err
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint="2080ti"

module load miniconda/22.11.1-1
# conda activate /work/pi_chuangg_umass_edu/yuncong/conda_envs/eqa-baseline
conda activate explore-eqa

python run_gpt_exp.py -cf cfg/eval_baseline_gpt.yaml