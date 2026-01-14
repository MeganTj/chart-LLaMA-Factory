#!/bin/bash
#SBATCH --job-name=bbox_train
#SBATCH --time=05:59:00
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=200G
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# slurm_single.sh - Submit single training job

# Create logs directory
mkdir -p logs

# Paths
export HF_HOME=/home/megantj/orcd/scratch/huggingface
export CONFIG_FILE=examples/train_chart/qwen3vl_lora_sft.yaml
export BASE_OUTPUT_DIR=/home/megantj/orcd/scratch/chartref_sft/qwen3vl/hyp_tune_saves

# Activate conda environment 
source ~/miniconda3/etc/profile.d/conda.sh
cd ~/orcd/scratch/projects/chart_compositional/chart-LLaMA-Factory; conda activate chartref2

# Run training
bash train_scripts/train_single.sh