.PHONY: install train-mnist train-cifar10 eval sample test clean lint format help

# Default target
help:
	@echo "Available commands:"
	@echo "  install       - Install dependencies"
	@echo "  train-mnist   - Train VAE on MNIST"
	@echo "  train-cifar10 - Train VAE on CIFAR-10"
	@echo "  eval          - Evaluate trained model"
	@echo "  sample        - Generate samples"
	@echo "  test          - Run test suite"
	@echo "  lint          - Run linting checks"
	@echo "  format        - Format code"
	@echo "  clean         - Clean outputs"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

# Training
train-mnist:
	python -m src.cli train --config configs/mnist.yaml

train-cifar10:
	python -m src.cli train --config configs/cifar10.yaml

# Evaluation and visualization
eval:
	python -m src.cli eval --config configs/mnist.yaml

sample:
	python -m src.cli sample prior --config configs/mnist.yaml

viz-recon:
	python -m src.cli viz recon_grid --config configs/mnist.yaml

viz-traverse:
	python -m src.cli viz traverse --config configs/mnist.yaml --dims 0 1 2

viz-scatter:
	python -m src.cli viz latent_scatter --config configs/mnist.yaml

interpolate:
	python -m src.cli sample interpolate --config configs/mnist.yaml

# Comparison with DAE
compare-dae:
	python -m src.cli compare dae --config configs/mnist.yaml --dae_ckpt ../Day_02_Denoising_Autoencoder/outputs/ckpts/best.pt

# Testing and code quality
test:
	pytest tests/ -v

lint:
	flake8 src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	isort src/ tests/

# Cleanup
clean:
	rm -rf outputs/ckpts/*.pt
	rm -rf outputs/logs/*
	rm -rf outputs/grids/*
	rm -rf outputs/samples/*
	rm -rf outputs/reports/*

clean-all: clean
	rm -rf data/
	rm -rf __pycache__/
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete