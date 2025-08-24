TARGETS := sgd-movement
TARGETS += adam-movement
TARGETS += adamw-movement
TARGETS += adamwadv-movement
HELP_TARGETS :=

all: $(TARGETS)

sgd-movement:
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

sgd-help-menu:
	@echo "sgd-movement:             - Run tests with SGD as a baseline"

HELP_TARGETS += sgd-help-menu

adam-movement:
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

adam-help-menu:
	@echo "adam-movement:            - Run tests with Adam as a baseline"

HELP_TARGETS += adam-help-menu

adamw-movement:
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

adamw-help-menu:
	@echo "adamw-movement:           - Run tests with AdamW as a baseline"

HELP_TARGETS += adamw-help-menu

adamwadv-movement:
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

adamwadv-help-menu:
	@echo "adamwadv-movement:        - Run tests with AdamW Advanced as a baseline"

HELP_TARGETS += adamwadv-help-menu

update-graphs:
	mv *.png images/

update-graphs-help-menu:
	@echo "update-graphs:            - Install new graphs to images/ dir"

HELP_TARGETS += update-graphs-help-menu

help: $(HELP_TARGETS)

clean:
	rm -f *.json
	rm -f *.png
