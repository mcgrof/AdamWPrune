# Top-level Makefile for AdamWPrune experiments
# SPDX-License-Identifier: MIT

# Default target MUST be first - declare it before any includes
.DEFAULT_GOAL := all

# Include Kconfig support to get CONFIG variables
include Makefile.kconfig

# Define what the default target does based on configuration
.PHONY: all
ifeq ($(CONFIG_OPTIMIZER_MODE_MULTIPLE),y)
all: check-config test-matrix
else
all: check-config memory-comparison update-graphs
endif

# Default model to train (can be overridden by Kconfig)
MODEL ?= $(if $(CONFIG_MODEL),$(CONFIG_MODEL),lenet5)

# Run memory comparison experiments using test matrix
memory-comparison: check-config
	@echo "Running memory comparison experiments..."
	@echo "Use 'make test-matrix' for comprehensive optimizer/pruning comparisons"
	@echo "Or configure specific tests with 'make menuconfig' then 'make train'"

# Update graphs with latest results
update-graphs: check-config generate-config
	@echo "Updating graphs from test matrix results..."
	@if [ -n "$(CONFIG_TEST_RESULTS_DIR)" ]; then \
		RESULTS_DIR="$(CONFIG_TEST_RESULTS_DIR)"; \
	else \
		RESULTS_DIR=$$(ls -d test_matrix_results_* 2>/dev/null | sort | tail -1); \
		if [ -z "$$RESULTS_DIR" ]; then \
			echo "Error: No test_matrix_results_* directories found"; \
			echo "Run 'make test-matrix' first or set TEST_RESULTS_DIR in menuconfig"; \
			exit 1; \
		fi; \
	fi; \
	echo "Using results from: $$RESULTS_DIR"; \
	python3 scripts/generate_optimizer_graphs.py "$$RESULTS_DIR" "$$RESULTS_DIR/graphs"; \
	mkdir -p images/lenet5; \
	echo "Copying graphs to images/lenet5/..."; \
	for optimizer in sgd adam adamw adamwadv adamwspam adamwprune; do \
		if [ -f "$$RESULTS_DIR/graphs/$${optimizer}_model_comparison.png" ]; then \
			cp "$$RESULTS_DIR/graphs/$${optimizer}_model_comparison.png" "images/lenet5/$${optimizer}_model_comparison.png"; \
			cp "$$RESULTS_DIR/graphs/$${optimizer}_accuracy_evolution.png" "images/lenet5/$${optimizer}_accuracy_evolution.png"; \
			echo "  Copied $$optimizer graphs"; \
		fi; \
	done; \
	echo "Graphs updated in images/lenet5/"

# Clean build artifacts but keep configuration
clean:
	@echo "Cleaning build artifacts (keeping configuration)..."
	@rm -f *.log train.log */train.log
	@rm -rf __pycache__
	@rm -rf */__pycache__
	@rm -rf lib/__pycache__
	@rm -rf scripts/__pycache__
	@rm -rf test_matrix_results_*
	@rm -f *.pyc *.pyo
	@rm -f training_metrics.json
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@if [ -d $(MODEL) ]; then $(MAKE) -C $(MODEL) clean; fi

# Clean everything including configuration and model files
# Returns workspace to pristine distribution state (keeps downloaded datasets)
mrproper: clean
	@echo "Removing all generated files, configuration, and model files..."
	@rm -f .config .config.old config.py
	@rm -f include/generated/autoconf.h
	@rm -rf include/config include/generated
	@rm -f *.pth
	@rm -f *.json
	@rm -f test-matrix.yaml
	@rm -f *.patch
	@rm -rf results/ */results/
	@rm -f *.tmp *.swp *~ */*~
	@rm -f *_backup.py */*_backup.py
	@if [ -d $(MODEL) ]; then $(MAKE) -C $(MODEL) mrproper 2>/dev/null || true; fi
	@echo "Workspace cleaned to pristine state (datasets preserved)."

# Clean downloaded datasets (use with caution - requires re-downloading)
data-clean:
	@echo "Removing downloaded datasets..."
	@rm -rf data/
	@rm -rf */data/
	@echo "Datasets removed. They will be re-downloaded on next training run."


# Train with current configuration
ifeq ($(CONFIG_TEST_MATRIX_MODE),y)
train: check-config test-matrix
else
train: check-config generate-config
	@echo "Training with configuration from .config..."
	cd $(MODEL) && python train.py --config ../config.py
endif

# Test matrix targets
test-matrix: check-config
	@echo "Running test matrix with configuration from .config..."
	@python3 scripts/run_test_matrix.py --config .config

test-matrix-yaml:
	@echo "Running test matrix with YAML configuration..."
	@python3 scripts/run_test_matrix.py --config-yaml test-matrix.yaml

test-matrix-dry-run: check-config
	@echo "Test matrix dry run (shows what would be executed)..."
	@python3 scripts/run_test_matrix.py --config .config --dry-run

# Re-run specific tests in an existing results directory
# Usage: make test-rerun TARGET=test_matrix_results_20250826_181029 [OPTIMIZER=adamwprune]
test-rerun:
	@if [ -z "$(TARGET)" ]; then \
		echo "Error: TARGET directory must be specified"; \
		echo "Usage: make test-rerun TARGET=test_matrix_results_YYYYMMDD_HHMMSS [OPTIMIZER=adamwprune]"; \
		echo ""; \
		echo "Available result directories:"; \
		ls -d test_matrix_results_* 2>/dev/null || echo "  (none found)"; \
		exit 1; \
	fi
	@if [ ! -d "$(TARGET)" ]; then \
		echo "Error: Directory '$(TARGET)' does not exist"; \
		echo ""; \
		echo "Available result directories:"; \
		ls -d test_matrix_results_* 2>/dev/null || echo "  (none found)"; \
		exit 1; \
	fi
	@echo "Re-running tests in: $(TARGET)"
	@if [ -n "$(OPTIMIZER)" ]; then \
		echo "Filtering to optimizer: $(OPTIMIZER)"; \
		python3 scripts/run_test_matrix.py \
			--config .config \
			--rerun-dir $(TARGET) \
			--filter-optimizer $(OPTIMIZER); \
	else \
		python3 scripts/run_test_matrix.py \
			--config .config \
			--rerun-dir $(TARGET); \
	fi

# Parallel execution targets
parallel: check-config
	@echo "Running test matrix with parallel execution..."
	@scripts/run_parallel_test_matrix.sh

parallel-4: check-config
	@echo "Running test matrix with 4 parallel jobs..."
	@scripts/run_parallel_test_matrix.sh -j 4

parallel-8: check-config
	@echo "Running test matrix with 8 parallel jobs..."
	@scripts/run_parallel_test_matrix.sh -j 8

parallel-16: check-config
	@echo "Running test matrix with 16 parallel jobs..."
	@scripts/run_parallel_test_matrix.sh -j 16

# Usage: make parallel-rerun TARGET=test_matrix_results_20250826_181029 [JOBS=8] [OPTIMIZER=adamwprune]
parallel-rerun:
	@if [ -z "$(TARGET)" ]; then \
		echo "Error: TARGET directory must be specified"; \
		echo "Usage: make parallel-rerun TARGET=test_matrix_results_YYYYMMDD_HHMMSS [JOBS=8] [OPTIMIZER=adamwprune]"; \
		echo ""; \
		echo "Available result directories:"; \
		ls -d test_matrix_results_* 2>/dev/null || echo "  (none found)"; \
		exit 1; \
	fi
	@JOBS=$${JOBS:-8}; \
	CMD_ARGS="-j $$JOBS -r $(TARGET)"; \
	if [ -n "$(OPTIMIZER)" ]; then \
		CMD_ARGS="$$CMD_ARGS -f $(OPTIMIZER)"; \
		echo "Re-running $(OPTIMIZER) tests in $(TARGET) with $$JOBS parallel jobs..."; \
	else \
		echo "Re-running all tests in $(TARGET) with $$JOBS parallel jobs..."; \
	fi; \
	scripts/run_parallel_test_matrix.sh $$CMD_ARGS

# Regenerate summary report from existing test results
summary:
	@python3 scripts/regenerate_summary.py

# Quick test matrix configurations
test-all-optimizers:
	@echo "Testing all optimizers with LeNet-5..."
	@cp defconfigs/test-matrix-optimizers .config
	@$(MAKE) test-matrix

test-all-pruning:
	@echo "Testing all pruning methods with LeNet-5..."
	@cp defconfigs/test-matrix-pruning .config
	@$(MAKE) test-matrix

test-everything:
	@echo "Testing all combinations (optimizers × pruning)..."
	@cp defconfigs/test-matrix-full .config
	@$(MAKE) test-matrix

# Help menu
help:
	@echo "AdamWPrune Experiments Makefile"
	@echo "================================"
	@echo ""
	@echo "Kconfig targets:"
	@echo "  menuconfig        - Configure using ncurses menu"
	@echo "  defconfig         - Load a default configuration (DEFCONFIG=name)"
	@echo "  defconfig-<tab>   - Tab-completable defconfig targets"
	@echo "  allyesconfig      - Enable all features (test matrix mode)"
	@echo "  allnoconfig       - Minimal configuration (SGD only)"
	@echo "  list-defconfigs   - List available default configurations"
	@echo "  savedefconfig     - Save current config as default"
	@echo "  kconfig-help      - Show all Kconfig targets"
	@echo ""
	@echo "Training targets:"
	@echo "  train             - Train with current configuration"
	@echo "  all               - Run memory comparison and update graphs (default)"
	@echo "  memory-comparison - Run all optimizer experiments with memory tracking"
	@echo "  update-graphs     - Update visualization graphs with latest results"
	@echo ""
	@echo "Cleaning targets:"
	@echo "  clean             - Clean build artifacts only (keeps config & datasets)"
	@echo "  mrproper          - Clean everything except datasets (removes config)"
	@echo "  data-clean        - Remove downloaded datasets (requires re-download)"
	@echo ""
	@echo "Test matrix targets:"
	@echo "  test-matrix       - Run test matrix from .config (serial)"
	@echo "  test-matrix-yaml  - Run test matrix from test-matrix.yaml"
	@echo "  test-matrix-dry-run - Show what would be tested without running"
	@echo "  test-rerun        - Re-run tests in existing directory"
	@echo "                      Usage: make test-rerun TARGET=<dir> [OPTIMIZER=<name>]"
	@echo "  summary           - Regenerate summary report from latest test results"
	@echo "  test-all-optimizers - Test all optimizers with LeNet-5"
	@echo "  test-all-pruning  - Test all pruning methods"
	@echo "  test-everything   - Test all combinations (optimizers × pruning)"
	@echo ""
	@echo "Parallel execution targets (for high-memory GPUs):"
	@echo "  parallel          - Run test matrix with parallel jobs (default: 8 jobs)"
	@echo "  parallel-4        - Run with 4 parallel jobs"
	@echo "  parallel-8        - Run with 8 parallel jobs (recommended for 48GB GPU)"
	@echo "  parallel-16       - Run with 16 parallel jobs"
	@echo "  parallel-rerun    - Re-run tests with parallel execution"
	@echo "                      Usage: make parallel-rerun TARGET=<dir> [JOBS=8] [OPTIMIZER=<name>]"
	@echo ""
	@echo "Quick start:"
	@echo "  make defconfig-lenet5           # Load LeNet-5 full config (tab-completable)"
	@echo "  make defconfig-lenet5-sgd       # Load LeNet-5 with SGD"
	@echo "  make defconfig-lenet5-adamwprune # Load LeNet-5 with AdamWPrune"
	@echo "  make menuconfig                 # Customize configuration"
	@echo "  make train                      # Train with current config"
	@echo ""
	@echo "Test matrix mode:"
	@echo "  make allyesconfig               # Configure for all combinations"
	@echo "  make test-matrix                # Run all test combinations (serial)"
	@echo "  make parallel                   # Run all test combinations (8 parallel jobs)"
	@echo "  make allyesconfig parallel-16   # Configure and run with 16 parallel jobs"
	@echo ""
	@echo "Parallel execution (recommended for GPUs with >24GB memory):"
	@echo "  make parallel                   # 8 jobs (good for 48GB GPU like W7900)"
	@echo "  make parallel-16                # 16 jobs (for very high memory GPUs)"
	@echo ""
	@echo "Re-running specific tests:"
	@echo "  make test-rerun TARGET=test_matrix_results_20250826_181029"
	@echo "                                  # Re-run all tests in existing directory"
	@echo "  make parallel-rerun TARGET=test_matrix_results_20250826_181029 OPTIMIZER=adamwprune"
	@echo "                                  # Re-run only adamwprune tests with parallel execution"
	@echo ""

.PHONY: all memory-comparison update-graphs clean mrproper data-clean help \
        train test-matrix test-matrix-yaml test-matrix-dry-run test-rerun summary \
        test-all-optimizers test-all-pruning test-everything \
        parallel parallel-4 parallel-8 parallel-16 parallel-rerun
