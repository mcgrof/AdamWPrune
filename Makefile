# Top-level Makefile for AdamWPrune experiments

# Default model to train
MODEL ?= lenet5

# Default target is to run memory comparison and update graphs
all: memory-comparison update-graphs

# Run memory comparison experiments for the current model
memory-comparison:
	@echo "Running memory comparison experiments for $(MODEL)..."
	$(MAKE) -C $(MODEL) memory-comparison

# Update graphs with latest results
update-graphs:
	@echo "Updating graphs for $(MODEL)..."
	$(MAKE) -C $(MODEL) update-graphs

# Clean all generated files
clean:
	@echo "Cleaning $(MODEL) directory..."
	$(MAKE) -C $(MODEL) clean

# Run specific optimizer experiments
sgd-movement:
	$(MAKE) -C $(MODEL) sgd-movement

adam-movement:
	$(MAKE) -C $(MODEL) adam-movement

adamw-movement:
	$(MAKE) -C $(MODEL) adamw-movement

adamwadv-movement:
	$(MAKE) -C $(MODEL) adamwadv-movement

adamwspam-movement:
	$(MAKE) -C $(MODEL) adamwspam-movement

adamwprune-movement:
	$(MAKE) -C $(MODEL) adamwprune-movement

sgd-magnitude:
	$(MAKE) -C $(MODEL) sgd-magnitude

adamw-magnitude:
	$(MAKE) -C $(MODEL) adamw-magnitude

# Help menu
help:
	@echo "AdamWPrune Experiments Makefile"
	@echo "================================"
	@echo ""
	@echo "Usage: make [target] [MODEL=model_name]"
	@echo ""
	@echo "Available models:"
	@echo "  lenet5            - LeNet-5 on MNIST (default)"
	@echo ""
	@echo "Main targets:"
	@echo "  all               - Run memory comparison and update graphs (default)"
	@echo "  memory-comparison - Run all optimizer experiments with memory tracking"
	@echo "  update-graphs     - Update visualization graphs with latest results"
	@echo "  clean            - Clean generated files"
	@echo ""
	@echo "Specific optimizer targets:"
	@echo "  sgd-movement      - Run SGD with movement pruning"
	@echo "  adam-movement     - Run Adam with movement pruning"
	@echo "  adamw-movement    - Run AdamW with movement pruning"
	@echo "  adamwadv-movement - Run AdamW Advanced with movement pruning"
	@echo "  adamwspam-movement- Run AdamW SPAM with movement pruning"
	@echo "  adamwprune-movement - Run AdamWPrune with state-based pruning"
	@echo "  sgd-magnitude     - Run SGD with magnitude pruning"
	@echo "  adamw-magnitude   - Run AdamW with magnitude pruning"
	@echo ""
	@echo "Examples:"
	@echo "  make                     # Run default (memory-comparison and update-graphs for lenet5)"
	@echo "  make MODEL=lenet5        # Explicitly specify lenet5 model"
	@echo "  make adamwprune-movement # Run AdamWPrune experiments only"

.PHONY: all memory-comparison update-graphs clean help \
        sgd-movement adam-movement adamw-movement adamwadv-movement \
        adamwspam-movement adamwprune-movement sgd-magnitude adamw-magnitude