#!/bin/bash
# Run the state pruning battle with clean output

echo "=================================================="
echo "ResNet-18 State Pruning Battle"
echo "=================================================="
echo "Configuration:"
echo "  - Optimizers: SGD, Adam, AdamWPrune (and others)"
echo "  - Pruning: State-based at 70% sparsity"
echo "  - GPU monitoring: Enabled"
echo "  - Inference testing: Enabled"
echo "=================================================="
echo ""

# Load the configuration (suppress warnings)
echo "Loading battle configuration..."
make defconfig-resnet18-state-pruning-battle 2>&1 | grep -v "warning:" | grep -v "make\[1\]"

# Verify key settings
echo ""
echo "Verifying configuration:"
grep -q "CONFIG_OPTIMIZER_MODE_MULTIPLE=y" .config && echo "  ✓ Multiple optimizers enabled"
grep -q "CONFIG_GPU_MONITOR=y" .config && echo "  ✓ GPU monitoring enabled"
grep -q "CONFIG_INFERENCE_TEST=y" .config && echo "  ✓ Inference testing enabled"
grep -q "CONFIG_PRUNING_SELECT_MOVEMENT=y" .config && echo "  ✓ Pruning configured (movement/state)"

echo ""
echo "Ready to start training!"
echo "Run 'make' to begin the state pruning battle"
echo ""
echo "The system will:"
echo "  1. Train each optimizer with state/movement pruning"
echo "  2. Monitor GPU memory during training"
echo "  3. Run inference tests with multiple batch sizes"
echo "  4. Monitor GPU memory during inference"
echo "  5. Automatically generate comparison graphs"
echo ""