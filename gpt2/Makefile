# GPT-2 Model Makefile

.PHONY: all clean data-clean

all:
	@echo "GPT-2 model - use parent Makefile for training"

clean:
	@rm -rf __pycache__
	@rm -f *.pyc *.pyo
	@rm -f train.log
	@rm -f training_metrics.json
	@rm -rf outputs/*.pt 2>/dev/null || true
	@rm -rf outputs/*.json 2>/dev/null || true

data-clean:
	@echo "Cleaning GPT-2 datasets..."
	@rm -rf data/shakespeare/*.bin 2>/dev/null || true
	@rm -rf data/shakespeare/*.txt 2>/dev/null || true
	@rm -rf data/finewebedu/*.bin 2>/dev/null || true
	@rm -rf data/openwebtext/*.bin 2>/dev/null || true
