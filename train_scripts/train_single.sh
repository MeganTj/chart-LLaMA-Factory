#!/bin/bash
# train_single.sh - Single training job (simplified)

# Hyperparameters with defaults
LR=${LR:-2e-5}
EPOCHS=${EPOCHS:-5}

# Paths with defaults
HF_HOME=${HF_HOME:-/home/megantj/orcd/scratch/huggingface}
CONFIG_FILE=${CONFIG_FILE:-examples/train_chart/qwen3vl_lora_sft.yaml}
BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-/home/megantj/orcd/scratch/chartref_sft/qwen3vl/hyp_tune_saves}

# Create run name
RUN_NAME="lr${LR}_ep${EPOCHS}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}"

echo "================================"
echo "Starting Training: $RUN_NAME"
echo "================================"
echo "Parameters:"
echo "  LR: $LR"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BS"
echo "  Output: $OUTPUT_DIR"
echo "  Time: $(date)"
echo "================================"

# Export HF_HOME
export HF_HOME=$HF_HOME

# Run training
llamafactory-cli train $CONFIG_FILE \
  learning_rate=$LR \
  num_train_epochs=$EPOCHS \
  run_name=$RUN_NAME \
  output_dir=$OUTPUT_DIR \

EXIT_CODE=$?

echo "================================"
echo "Training Completed: $RUN_NAME"
echo "Exit Code: $EXIT_CODE"
echo "Time: $(date)"
echo "================================"

exit $EXIT_CODE