#!/bin/bash
# Launch script for DDP training on multi-GPU systems
# Usage: ./gpt2/launch_ddp.sh [num_gpus]

NUM_GPUS=${1:-8}  # Default to 8 GPUs for H100x8

echo "Launching DDP training on $NUM_GPUS GPUs..."
echo "================================"

# Use torchrun for launching (modern PyTorch distributed launcher)
torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    gpt2/train.py \
    "${@:2}"  # Pass any additional arguments

# Alternative using python -m torch.distributed.launch (older method):
# python -m torch.distributed.launch \
#     --nproc_per_node=$NUM_GPUS \
#     --use_env \
#     gpt2/train.py \
#     "${@:2}"
