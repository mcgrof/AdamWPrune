# LeNet-5 Model Makefile
# SPDX-License-Identifier: MIT
#
# This Makefile is now legacy - use the main Makefile with Kconfig instead:
#   make menuconfig
#   make train
#   make test-matrix

.PHONY: all clean mrproper help

# Default target warns about legacy usage
all:
	@echo "WARNING: This Makefile is legacy and deprecated."
	@echo ""
	@echo "Use the modern Kconfig-based system instead:"
	@echo "  cd .. && make menuconfig        # Configure"
	@echo "  cd .. && make train             # Train model"
	@echo "  cd .. && make test-matrix       # Run test matrix"
	@echo ""
	@echo "The new system is more flexible and supports all optimizers/pruning methods."

# Clean build artifacts
clean:
	@echo "Cleaning LeNet-5 build artifacts..."
	@rm -f *.pth *.log *.json
	@rm -rf __pycache__
	@find . -name "*.pyc" -delete
	@find . -name "*.pyo" -delete

# Clean everything
mrproper: clean
	@echo "LeNet-5 cleaned to pristine state"

# Legacy memory comparison (redirects to new system)
memory-comparison:
	@echo "Redirecting to new Kconfig-based system..."
	@cd .. && $(MAKE) memory-comparison

help:
	@echo "LeNet-5 Legacy Makefile (DEPRECATED)"
	@echo "====================================="
	@echo ""
	@echo "This Makefile is deprecated. Use the main Makefile instead:"
	@echo ""
	@echo "  cd .. && make help              # Show all available options"
	@echo "  cd .. && make menuconfig        # Interactive configuration"
	@echo "  cd .. && make list-defconfigs   # Show available presets"
	@echo "  cd .. && make train             # Train with current config"
	@echo "  cd .. && make test-matrix       # Run full test matrix"
	@echo ""
	@echo "Local targets:"
	@echo "  clean                           # Clean build artifacts"
	@echo "  mrproper                        # Clean everything"
