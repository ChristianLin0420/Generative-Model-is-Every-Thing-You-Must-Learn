# Makefile for Forward Diffusion Process

.PHONY: help install test clean lint format run-mnist run-cifar10 run-all

# Default target
help:
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run test suite"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  clean       - Clean outputs and cache"
	@echo "  run-mnist   - Run all experiments on MNIST"
	@echo "  run-cifar10 - Run all experiments on CIFAR-10"
	@echo "  run-all     - Run experiments on both datasets"
	@echo "  figures     - Generate specific figure bundle"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

install-dev: install
	pip install pytest black flake8 isort

# Testing
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	isort --check-only src/ tests/

format:
	black src/ tests/ --line-length=100
	isort src/ tests/

# Cleaning
clean:
	rm -rf outputs/grids/*
	rm -rf outputs/animations/*
	rm -rf outputs/plots/*
	rm -rf outputs/logs/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-all: clean
	rm -rf data/
	rm -rf .pytest_cache/
	rm -rf htmlcov/

# Experiments
run-mnist:
	./scripts/run_mnist.sh

run-cifar10:
	./scripts/run_cifar10.sh

run-all: run-mnist run-cifar10

# Generate specific figures
figures:
	@echo "Usage: make figures DATASET=mnist|cifar10"
	@if [ -z "$(DATASET)" ]; then \
		echo "Please specify DATASET=mnist or DATASET=cifar10"; \
		exit 1; \
	fi
	./scripts/make_figures.sh $(DATASET)

# Quick experiments
quick-mnist:
	python -m src.cli traj.grid --config configs/mnist.yaml
	python -m src.cli plots.schedules --config configs/mnist.yaml

quick-cifar10:
	python -m src.cli traj.grid --config configs/cifar10.yaml
	python -m src.cli plots.schedules --config configs/cifar10.yaml

# Individual commands
trajectory-grid:
	python -m src.cli traj.grid --config configs/$(DATASET).yaml

trajectory-animate:
	python -m src.cli traj.animate --config configs/$(DATASET).yaml --idx $(IDX)

plot-schedules:
	python -m src.cli plots.schedules --config configs/$(DATASET).yaml

plot-snr:
	python -m src.cli plots.snr --config configs/$(DATASET).yaml

plot-histograms:
	python -m src.cli plots.hist --config configs/$(DATASET).yaml

compute-stats:
	python -m src.cli compute.stats --config configs/$(DATASET).yaml

# Development helpers
dev-setup: install-dev
	pre-commit install || echo "pre-commit not available"

check: lint test

# Jupyter notebook
notebook:
	jupyter lab notebooks/04_forward_diffusion.ipynb

# Documentation
docs:
	@echo "Forward Diffusion Process Documentation"
	@echo "======================================"
	@cat README.md

# Examples
examples:
	@echo "Example commands:"
	@echo ""
	@echo "Basic usage:"
	@echo "  make run-mnist"
	@echo "  make run-cifar10"
	@echo ""
	@echo "Individual experiments:"
	@echo "  make trajectory-grid DATASET=mnist"
	@echo "  make trajectory-animate DATASET=cifar10 IDX=5"
	@echo "  make plot-schedules DATASET=mnist"
	@echo ""
	@echo "Development:"
	@echo "  make test"
	@echo "  make lint"
	@echo "  make clean"