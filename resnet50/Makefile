# SPDX-License-Identifier: MIT
# ResNet-50 training Makefile

-include ../.config

# Model-specific variables
MODEL_NAME := resnet50
DATASET := $(CONFIG_RESNET50_DATASET)
DATA_DIR := $(CONFIG_RESNET50_DATA_DIR)
BATCH_SIZE := $(CONFIG_RESNET50_BATCH_SIZE)
LEARNING_RATE := $(CONFIG_RESNET50_LEARNING_RATE)
EPOCHS := $(CONFIG_RESNET50_EPOCHS)
NUM_WORKERS := $(CONFIG_RESNET50_NUM_WORKERS)
LOG_INTERVAL := $(CONFIG_RESNET50_LOG_INTERVAL)

# Training arguments
TRAIN_ARGS := --dataset $(DATASET) \
              --data-dir $(DATA_DIR) \
              --batch-size $(BATCH_SIZE) \
              --epochs $(EPOCHS) \
              --lr $(LEARNING_RATE) \
              --optimizer $(OPTIMIZER) \
              --pruning-method $(PRUNING_METHOD) \
              --target-sparsity $(TARGET_SPARSITY) \
              --num-workers $(NUM_WORKERS) \
              --log-interval $(LOG_INTERVAL)

# Optional arguments
ifdef CONFIG_RESNET50_SAVE_MODEL
TRAIN_ARGS += --save-model
endif

ifdef CONFIG_RESNET50_MONITOR_GPU
TRAIN_ARGS += --monitor-gpu
endif

ifdef CONFIG_RESNET50_PRUNING_START_EPOCH
TRAIN_ARGS += --pruning-start-epoch $(CONFIG_RESNET50_PRUNING_START_EPOCH)
endif

ifdef CONFIG_RESNET50_PRUNING_END_EPOCH
TRAIN_ARGS += --pruning-end-epoch $(CONFIG_RESNET50_PRUNING_END_EPOCH)
endif

# Output directory
OUTPUT_DIR := $(TEST_ID)

.PHONY: all train clean test-model

all: train

train:
	@echo "Training ResNet-50 with configuration:"
	@echo "  Model: ResNet-50"
	@echo "  Dataset: $(DATASET)"
	@echo "  Optimizer: $(OPTIMIZER)"
	@echo "  Pruning: $(PRUNING_METHOD) ($(TARGET_SPARSITY))"
	@echo "  Batch Size: $(BATCH_SIZE)"
	@echo "  Learning Rate: $(LEARNING_RATE)"
	@echo "  Epochs: $(EPOCHS)"
	@mkdir -p $(OUTPUT_DIR)
	@cd $(OUTPUT_DIR) && python ../train.py $(TRAIN_ARGS) 2>&1 | tee training.log
	@echo "Training complete. Results saved in $(OUTPUT_DIR)/"

test-model:
	@echo "Testing ResNet-50 model..."
	@python model.py

clean:
	rm -rf resnet50_*/ __pycache__/ *.pth *.json *.log

help:
	@echo "ResNet-50 Training Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  train       - Train the model with current configuration"
	@echo "  test-model  - Test model creation and parameter count"
	@echo "  clean       - Remove all generated files"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Current Configuration:"
	@echo "  Dataset: $(DATASET)"
	@echo "  Optimizer: $(OPTIMIZER)"
	@echo "  Pruning: $(PRUNING_METHOD) at $(TARGET_SPARSITY) sparsity"
	@echo ""
	@echo "To change configuration, run 'make menuconfig' in parent directory"