# ChartRef SFT

Codebase for finetuning MLLMs on ChartRef

## Conda environment
Run the following commands
```
conda create --name chartref python=3.11
pip install uv
# Step 1: Install PyTorch with support for the specific CUDA version
uv pip install vllm --torch-backend=cu124
pip install google.generativeai qwen_vl_utils anthropic nvitop
pip install wandb PyPDF2 pdf2image pandas
pip install flash-attn==2.7.4.post1 --no-build-isolation

# Install LLaMA-Factory
cd chart-LLaMA-Factory
pip install -e ".[torch,metrics]"
```

## SFT Pipeline

Unzip the `images` directory and move `chartref_sft` to `data`.

Set the `output_dir`in the yaml file to your local directory.
```
llamafactory-cli train examples/train_chart/qwen3vl_lora_sft.yaml
```

Set the `adapter_name_or_path` and `export_dir` in the yaml file to your local directory.
```
llamafactory-cli export examples/merge_lora/chart_qwen3vl_lora_sft.yaml 
```

Evaluate performance on test set
```
python scripts/vllm_infer.py --model_name_or_path /home/megantj/orcd/scratch/chartref_sft/qwen3vl/output  --template qwen3_vl --dataset qwen3vl_chartref_test  --save_name "qwen3vl_generated_predictions_test.jsonl"

python scripts/eval_chartref.py  --filename "qwen3vl_generated_predictions_test.jsonl"
```