#!/bin/bash
# submit_sweep.sh - Submit multiple jobs using slurm_single.sh

# Create logs directory
mkdir -p logs

# Define hyperparameters to sweep
learning_rates=(1e-5 5e-5 1e-4 5e-4)
num_epochs=(2 3 5)

# Submit each combination
for lr in "${learning_rates[@]}"; do
  for epochs in "${num_epochs[@]}"; do
    
    # Create unique job name
    JOB_NAME="bbox_lr${lr}_ep${epochs}"
    
    echo "Submitting: $JOB_NAME"
    
    # Submit slurm_single.sh with env vars (BS will use default)
    sbatch \
      --job-name=$JOB_NAME \
      --export=ALL,LR=$lr,EPOCHS=$epochs \
      train_scripts/slurm_single.sh
    
    sleep 0.5
    
  done
done

echo "All 12 jobs submitted! Check status with: squeue -u $USER"