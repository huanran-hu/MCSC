#!/bin/bash
# run.sh
# ======
# Launch full-parameter fine-tuning on 8x A100-80GB.
#
# Usage:
#   bash run.sh
#
# Prerequisites:
#   pip install transformers==4.57.1 torch>=2.6.0 deepspeed>=0.16.0 \
#               safetensors pyyaml flash-attn tensorboard

set -euo pipefail

# ---- Config ----
CONFIG="config.yaml"
DS_CONFIG="ds_config.json"
NUM_GPUS=8
MASTER_PORT=29500

echo "============================================="
echo "  Qwen3-VL Fine-tuning (Frozen ViT)"
echo "  Config:    ${CONFIG}"
echo "  DeepSpeed: ${DS_CONFIG}"
echo "  GPUs:      ${NUM_GPUS}"
echo "============================================="

# ---- Launch with DeepSpeed ----
deepspeed \
    --num_gpus=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    train.py \
    --config ${CONFIG} \
    --deepspeed ${DS_CONFIG}

echo "Training completed!"
