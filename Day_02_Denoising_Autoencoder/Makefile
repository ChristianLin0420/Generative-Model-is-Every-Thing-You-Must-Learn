.PHONY: setup install train eval test viz clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  setup      - Create virtual environment and install dependencies"
	@echo "  install    - Install dependencies only"
	@echo "  train      - Train DAE on MNIST (default)"
	@echo "  eval       - Evaluate trained model"
	@echo "  viz        - Generate all visualizations"
	@echo "  test       - Run tests"
	@echo "  clean      - Clean outputs and cache"
	@echo "  help       - Show this help message"

setup:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt

install:
	pip install -r requirements.txt

# Training targets
train:
	python -m src.cli train --config configs/mnist.yaml

train-quick:
	python -m src.cli train --config configs/mnist.yaml --epochs 5

# Evaluation targets
eval:
	python -m src.cli eval --config configs/mnist.yaml

# Visualization targets
viz:
	python -m src.cli viz.recon_grid --config configs/mnist.yaml
	python -m src.cli viz.sigma_panel --config configs/mnist.yaml

viz-all:
	./scripts/make_figures.sh

# Analysis targets
compare:
	python -m src.cli compare.limitations --config configs/mnist.yaml

# Development targets
test:
	pytest tests/ -v

test-quick:
	pytest tests/ -v -x

lint:
	black src/ tests/
	isort src/ tests/
	ruff check src/ tests/

clean:
	rm -rf outputs/ckpts/*
	rm -rf outputs/logs/*
	rm -rf outputs/grids/*
	rm -rf outputs/panels/*
	rm -rf outputs/reports/*
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Quick development workflows
demo-mnist:
	chmod +x scripts/train_mnist.sh
	./scripts/train_mnist.sh

figures:
	chmod +x scripts/make_figures.sh
	./scripts/make_figures.sh

# Full pipeline
full-pipeline:
	make train
	make eval
	make viz
	make compare