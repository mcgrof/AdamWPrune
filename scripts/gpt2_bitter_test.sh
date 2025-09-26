#!/bin/bash
# Quick GPT-2 test for bitter3 and bitter4 variants

echo "===================================="
echo "Testing bitter3 variant with GPT-2"
echo "===================================="

cd /data/AdamWPrune/gpt2

# Test bitter3 with minimal iterations
timeout 30 python train.py \
    --optimizer adamwprune \
    --adamwprune-variant bitter3 \
    --pruning-method state \
    --target-sparsity 0.5 \
    --pruning-warmup 5 \
    --batch-size 2 \
    --gradient-accumulation 2 \
    --max-iters 20 \
    --eval-interval 10 \
    --eval-samples 10 \
    --device cuda \
    --log-interval 5 \
    --dataset shakespeare \
    --block-size 256 \
    --model-name gpt2 \
    2>&1 | tail -20

echo ""
echo "===================================="
echo "Testing bitter4 variant with GPT-2"
echo "===================================="

# Test bitter4 with minimal iterations
timeout 30 python train.py \
    --optimizer adamwprune \
    --adamwprune-variant bitter4 \
    --pruning-method state \
    --target-sparsity 0.5 \
    --pruning-warmup 5 \
    --batch-size 2 \
    --gradient-accumulation 2 \
    --max-iters 20 \
    --eval-interval 10 \
    --eval-samples 10 \
    --device cuda \
    --log-interval 5 \
    --dataset shakespeare \
    --block-size 256 \
    --model-name gpt2 \
    2>&1 | tail -20

echo ""
echo "✓ Quick GPT-2 test completed"
