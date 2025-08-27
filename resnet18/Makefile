# SPDX-License-Identifier: MIT
# Makefile for ResNet-18 model

.PHONY: all train clean

# Default target
all: train

# Train ResNet-18
train:
	@echo "Training ResNet-18 on CIFAR-10..."
	@python3 train.py

# Clean generated files
clean:
	@echo "Cleaning ResNet-18 build artifacts..."
	@rm -rf __pycache__ *.pyc *.pyo
	@rm -f training_metrics.json resnet18_checkpoint.pth
	# Note: data/ directory preserved to avoid re-downloading CIFAR-10 (170MB)

# Test the model implementation
test:
	@echo "Testing ResNet-18 model..."
	@python3 model.py

.PHONY: help
help:
	@echo "ResNet-18 Makefile targets:"
	@echo "  make train  - Train ResNet-18 model"
	@echo "  make test   - Test model implementation"
	@echo "  make clean  - Clean build artifacts"
