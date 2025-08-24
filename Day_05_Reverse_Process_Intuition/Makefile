# Makefile for Day 5: Reverse Process Intuition

.PHONY: install clean train-mnist train-cifar10 sample test lint format

# Install dependencies
install:
	pip install -r requirements.txt

# Clean outputs
clean:
	rm -rf outputs/ckpts/*
	rm -rf outputs/logs/*
	rm -rf outputs/grids/*
	rm -rf outputs/animations/*
	rm -rf outputs/samples/*
	rm -rf outputs/reports/*

# Training
train-mnist:
	python -m src.cli train --config configs/mnist.yaml

train-cifar10:
	python -m src.cli train --config configs/cifar10.yaml

# Quick training (fewer epochs)
train-mnist-quick:
	python -m src.cli train --config configs/mnist.yaml --epochs 5

train-cifar10-quick:
	python -m src.cli train --config configs/cifar10.yaml --epochs 5

# Sampling
sample-grid:
	python -m src.cli sample.grid --ckpt outputs/ckpts/model_final.pth --num_samples 64

sample-traj:
	python -m src.cli sample.traj --ckpt outputs/ckpts/model_final.pth --num_samples 4

# Visualization
viz-compare:
	python -m src.cli viz.compare --ckpt outputs/ckpts/model_final.pth

# Evaluation
eval:
	python -m src.cli eval --ckpt outputs/ckpts/model_final.pth

# All figures
figures: sample-grid sample-traj viz-compare

# Testing
test:
	python -m pytest tests/ -v

# Code quality
lint:
	python -m flake8 src/ tests/
	python -m black --check src/ tests/

format:
	python -m black src/ tests/
	python -m isort src/ tests/

# Complete pipeline
all: train-mnist-quick figures eval

# Help
help:
	@echo "Available targets:"
	@echo "  install       - Install dependencies"
	@echo "  clean         - Clean output directories"
	@echo "  train-mnist   - Train on MNIST dataset"
	@echo "  train-cifar10 - Train on CIFAR-10 dataset"
	@echo "  sample-grid   - Generate grid of samples"
	@echo "  sample-traj   - Generate trajectory samples"
	@echo "  viz-compare   - Create forward vs reverse comparison"
	@echo "  eval          - Run evaluation metrics"
	@echo "  test          - Run test suite"
	@echo "  lint          - Check code quality"
	@echo "  format        - Format code"
	@echo "  all           - Run complete pipeline"
	@echo "  help          - Show this help"