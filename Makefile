all: sgd-movement

sgd-movement:
	# Model A - Baseline (no pruning)
	python train.py
	cp training_metrics.json model_a_metrics.json

	# Model B - 50% pruning
	python train.py --pruning-method movement --target-sparsity 0.5
	cp training_metrics.json model_b_metrics.json

	# Model C - 90% pruning
	python train.py --pruning-method movement --target-sparsity 0.9
	cp training_metrics.json model_c_metrics.json

	# Model D - 70% pruning
	python train.py --pruning-method movement --target-sparsity 0.7
	cp training_metrics.json model_d_metrics.json

	# Generate comparison plots
	python plot_comparison.py

clean:
	rm -f *.json
