TARGETS := sgd-movement
TARGETS += adam-movement
TARGETS += adamw-movement
TARGETS += adamwadv-movement
TARGETS += adamwspam-movement
TARGETS += adamwprune-movement
HELP_TARGETS :=

all: $(TARGETS)

sgd-movement:
	@mkdir -p results/sgd
	# Model A - Baseline (no pruning)
	python train.py --json-output model_a_metrics.json

	# Model B - 50% pruning
	python train.py --pruning-method movement --target-sparsity 0.5 \
		--json-output model_b_metrics.json

	# Model C - 90% pruning
	python train.py --pruning-method movement --target-sparsity 0.9 \
		--json-output model_c_metrics.json

	# Model D - 70% pruning
	python train.py --pruning-method movement --target-sparsity 0.7 \
		--json-output model_d_metrics.json

	# Generate comparison plots
	python plot_comparison.py \
		--compare-output sgd-with-movement-accuracy_evolution.png \
		--accuracy-output sgd-with-movement-pruning-model_comparison.png

	# Save results
	@mv model_*.json results/sgd/

sgd-help-menu:
	@echo "sgd-movement:             - Run tests with SGD as a baseline"

HELP_TARGETS += sgd-help-menu

adam-movement:
	@mkdir -p results/adam
	# Model A - Baseline (no pruning)
	python train.py --optimizer adam --json-output model_a_metrics.json

	# Model B - 50% pruning
	python train.py --optimizer adam --pruning-method movement \
		--target-sparsity 0.5 \
		--json-output model_b_metrics.json

	# Model C - 90% pruning
	python train.py --optimizer adam --pruning-method movement \
		--target-sparsity 0.9 \
		--json-output model_c_metrics.json

	# Model D - 70% pruning
	python train.py --optimizer adam --pruning-method movement \
		--target-sparsity 0.7 \
		--json-output model_d_metrics.json

	# Generate comparison plots
	python plot_comparison.py --test-prefix="Adam" \
		--compare-output adam-with-movement-accuracy_evolution.png \
		--accuracy-output adam-with-movement-pruning-model_comparison.png

	# Save results
	@mv model_*.json results/adam/

adam-help-menu:
	@echo "adam-movement:            - Run tests with Adam as a baseline"

HELP_TARGETS += adam-help-menu

adamw-movement:
	@mkdir -p results/adamw
	# Model A - Baseline (no pruning)
	python train.py --optimizer adamw --json-output model_a_metrics.json

	# Model B - 50% pruning
	python train.py --optimizer adamw --pruning-method movement \
		--target-sparsity 0.5 \
		--json-output model_b_metrics.json

	# Model C - 90% pruning
	python train.py --optimizer adamw --pruning-method movement \
		--target-sparsity 0.9 \
		--json-output model_c_metrics.json

	# Model D - 70% pruning
	python train.py --optimizer adamw --pruning-method movement \
		--target-sparsity 0.7 \
		--json-output model_d_metrics.json

	# Generate comparison plots
	python plot_comparison.py --test-prefix="AdamW" \
		--compare-output adamw-with-movement-accuracy_evolution.png \
		--accuracy-output adamw-with-movement-pruning-model_comparison.png

	# Save results
	@mv model_*.json results/adamw/

adamw-help-menu:
	@echo "adamw-movement:           - Run tests with AdamW as a baseline"

HELP_TARGETS += adamw-help-menu

adamwadv-movement:
	@mkdir -p results/adamwadv
	# Model A - Baseline (no pruning)
	python train.py --optimizer adamwadv --json-output model_a_metrics.json

	# Model B - 50% pruning
	python train.py --optimizer adamwadv --pruning-method movement \
		--target-sparsity 0.5 \
		--json-output model_b_metrics.json

	# Model C - 90% pruning
	python train.py --optimizer adamwadv --pruning-method movement \
		--target-sparsity 0.9 \
		--json-output model_c_metrics.json

	# Model D - 70% pruning
	python train.py --optimizer adamwadv --pruning-method movement \
		--target-sparsity 0.7 \
		--json-output model_d_metrics.json

	# Generate comparison plots
	python plot_comparison.py --test-prefix="AdamWAdv" \
		--compare-output adamwadv-with-movement-accuracy_evolution.png \
		--accuracy-output adamwadv-with-movement-pruning-model_comparison.png

	# Save results
	@mv model_*.json results/adamwadv/

adamwadv-help-menu:
	@echo "adamwadv-movement:        - Run tests with AdamW Advanced as a baseline"

HELP_TARGETS += adamwadv-help-menu

adamwspam-movement:
	@mkdir -p results/adamwspam
	# Model A - Baseline (no pruning)
	python train.py --optimizer adamwspam --json-output model_a_metrics.json

	# Model B - 50% pruning
	python train.py --optimizer adamwspam --pruning-method movement \
		--target-sparsity 0.5 \
		--json-output model_b_metrics.json

	# Model C - 90% pruning
	python train.py --optimizer adamwspam --pruning-method movement \
		--target-sparsity 0.9 \
		--json-output model_c_metrics.json

	# Model D - 70% pruning
	python train.py --optimizer adamwspam --pruning-method movement \
		--target-sparsity 0.7 \
		--json-output model_d_metrics.json

	# Generate comparison plots
	python plot_comparison.py --test-prefix="AdamWSPAM" \
		--compare-output adamwspam-with-movement-accuracy_evolution.png \
		--accuracy-output adamwspam-with-movement-pruning-model_comparison.png

	# Save results
	@mv model_*.json results/adamwspam/

adamwspam-help-menu:
	@echo "adamwspam-movement:       - Run tests with AdamW SPAM as a baseline"

HELP_TARGETS += adamwspam-help-menu

adamwprune-movement:
	@mkdir -p results/adamwprune
	# Model A - Baseline (no pruning)
	python train.py --optimizer adamwprune --json-output model_a_metrics.json

	# Model B - 50% pruning (uses state-based pruning)
	python train.py --optimizer adamwprune --pruning-method movement \
		--target-sparsity 0.5 \
		--json-output model_b_metrics.json

	# Model C - 90% pruning (uses state-based pruning)
	python train.py --optimizer adamwprune --pruning-method movement \
		--target-sparsity 0.9 \
		--json-output model_c_metrics.json

	# Model D - 70% pruning (uses state-based pruning)
	python train.py --optimizer adamwprune --pruning-method movement \
		--target-sparsity 0.7 \
		--json-output model_d_metrics.json

	# Generate comparison plots
	python plot_comparison.py --test-prefix="AdamWPrune" \
		--compare-output adamwprune-with-movement-accuracy_evolution.png \
		--accuracy-output adamwprune-with-movement-pruning-model_comparison.png

	# Save results
	@mv model_*.json results/adamwprune/

adamwprune-help-menu:
	@echo "adamwprune-movement:      - Run tests with AdamWPrune (experimental state-based pruning)"

HELP_TARGETS += adamwprune-help-menu

# Magnitude pruning targets (baseline pruning method)
sgd-magnitude:
	@mkdir -p results/sgd-magnitude
	# Model A - Baseline (no pruning)
	python train.py --json-output model_a_metrics.json

	# Model B - 50% pruning
	python train.py --pruning-method magnitude --target-sparsity 0.5 \
		--json-output model_b_metrics.json

	# Model C - 90% pruning
	python train.py --pruning-method magnitude --target-sparsity 0.9 \
		--json-output model_c_metrics.json

	# Model D - 70% pruning
	python train.py --pruning-method magnitude --target-sparsity 0.7 \
		--json-output model_d_metrics.json

	# Generate comparison plots
	python plot_comparison.py \
		--compare-output sgd-with-magnitude-accuracy_evolution.png \
		--accuracy-output sgd-with-magnitude-pruning-model_comparison.png

	# Save results
	@mv model_*.json results/sgd-magnitude/

sgd-magnitude-help-menu:
	@echo "sgd-magnitude:            - Run tests with SGD and magnitude pruning"

HELP_TARGETS += sgd-magnitude-help-menu

adamw-magnitude:
	@mkdir -p results/adamw-magnitude
	# Model A - Baseline (no pruning)
	python train.py --optimizer adamw --json-output model_a_metrics.json

	# Model B - 50% pruning
	python train.py --optimizer adamw --pruning-method magnitude \
		--target-sparsity 0.5 \
		--json-output model_b_metrics.json

	# Model C - 90% pruning
	python train.py --optimizer adamw --pruning-method magnitude \
		--target-sparsity 0.9 \
		--json-output model_c_metrics.json

	# Model D - 70% pruning
	python train.py --optimizer adamw --pruning-method magnitude \
		--target-sparsity 0.7 \
		--json-output model_d_metrics.json

	# Generate comparison plots
	python plot_comparison.py \
		--compare-output adamw-with-magnitude-accuracy_evolution.png \
		--accuracy-output adamw-with-magnitude-pruning-model_comparison.png

	# Save results
	@mv model_*.json results/adamw-magnitude/

adamw-magnitude-help-menu:
	@echo "adamw-magnitude:          - Run tests with AdamW and magnitude pruning"

HELP_TARGETS += adamw-magnitude-help-menu

update-graphs:
	mv *.png images/

update-graphs-help-menu:
	@echo "update-graphs:            - Install new graphs to images/ dir"

HELP_TARGETS += update-graphs-help-menu

# Memory comparison visualization targets
memory-comparison: sgd-movement adam-movement adamw-movement adamwadv-movement adamwspam-movement adamwprune-movement sgd-magnitude adamw-magnitude
	@echo "Generating memory comparison visualizations..."
	python plot_optimizer_memory_comparison.py
	@echo "Memory comparison plots generated:"
	@echo "  - optimizer_comparison_baseline.png"
	@echo "  - optimizer_comparison_50_pruning.png"
	@echo "  - optimizer_comparison_70_pruning.png"
	@echo "  - optimizer_comparison_90_pruning.png"
	@echo "  - memory_efficiency_summary.png"

memory-comparison-help-menu:
	@echo "memory-comparison:        - Generate memory efficiency comparison plots"

HELP_TARGETS += memory-comparison-help-menu

help: $(HELP_TARGETS)

clean:
	rm -f *.json
	rm -f *.png
