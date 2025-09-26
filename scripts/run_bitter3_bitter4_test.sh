#!/bin/bash
# Run bitter3 and bitter4 tests directly

echo "=================================="
echo "Running AdamWPrune bitter3/bitter4 tests"
echo "=================================="

cd /data/AdamWPrune/gpt2

# Test parameters (short test)
MAX_ITERS="${MAX_ITERS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25}"

echo "Test configuration:"
echo "  Max iterations: $MAX_ITERS"
echo "  Batch size: $BATCH_SIZE"
echo "  Eval interval: $EVAL_INTERVAL"
echo ""

# Run bitter3
echo "=================================="
echo "Testing bitter3 variant"
echo "=================================="
python train.py \
    --optimizer adamwprune \
    --adamwprune-variant bitter3 \
    --pruning-method state \
    --target-sparsity 0.5 \
    --pruning-warmup 10 \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation 2 \
    --max-iters $MAX_ITERS \
    --eval-interval $EVAL_INTERVAL \
    --eval-samples 50 \
    --log-interval 10 \
    --device cuda \
    --dataset finewebedu \
    --output-dir bitter3_test_$MAX_ITERS \
    --tracker wandb,trackio

echo ""
echo "=================================="
echo "Testing bitter4 variant"
echo "=================================="
python train.py \
    --optimizer adamwprune \
    --adamwprune-variant bitter4 \
    --pruning-method state \
    --target-sparsity 0.5 \
    --pruning-warmup 10 \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation 2 \
    --max-iters $MAX_ITERS \
    --eval-interval $EVAL_INTERVAL \
    --eval-samples 50 \
    --log-interval 10 \
    --device cuda \
    --dataset finewebedu \
    --output-dir bitter4_test_$MAX_ITERS \
    --tracker wandb,trackio

echo ""
echo "=================================="
echo "Tests completed!"
echo "=================================="