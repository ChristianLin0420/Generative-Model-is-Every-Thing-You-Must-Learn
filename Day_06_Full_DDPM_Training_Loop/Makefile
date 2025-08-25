.PHONY: setup install train sample eval test lint clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  setup     - Install dependencies and setup environment"
	@echo "  install   - Install package in development mode"
	@echo "  train     - Train DDPM on MNIST (default)"
	@echo "  train-cifar - Train DDPM on CIFAR-10"
	@echo "  sample    - Generate sample grid (64 images)"
	@echo "  eval      - Evaluate trained model"
	@echo "  test      - Run test suite"
	@echo "  lint      - Run linting and formatting"
	@echo "  clean     - Clean generated files"

# Setup and installation
setup:
	pip install -r requirements.txt
	pip install -e .

install: setup

# Training commands
train:
	python -m src.cli train --config configs/mnist.yaml

train-cifar:
	python -m src.cli train --config configs/cifar10.yaml

train-mnist:
	bash scripts/train_mnist.sh

train-cifar10:
	bash scripts/train_cifar10.sh

# Sampling commands
sample:
	python -m src.cli sample.grid --config configs/mnist.yaml --num_samples 64

sample-grid:
	bash scripts/sample_64.sh

sample-traj:
	bash scripts/animate_traj.sh

# Evaluation
eval:
	python -m src.cli eval --config configs/mnist.yaml

eval-cifar:
	python -m src.cli eval --config configs/cifar10.yaml

# Visualization
viz-curves:
	python -m src.cli viz.curves --log_dir outputs/logs

viz-schedules:
	python -m src.cli viz.schedules --config configs/mnist.yaml

# Development
test:
	pytest -q tests/

test-verbose:
	pytest -v tests/

lint:
	black src/ tests/
	isort src/ tests/
	flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503

# Cleanup
clean:
	rm -rf outputs/ckpts/*
	rm -rf outputs/logs/*
	rm -rf outputs/grids/*
	rm -rf outputs/animations/*
	rm -rf outputs/curves/*
	rm -rf outputs/reports/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Generate comprehensive report
report:
	bash scripts/make_report.sh

# Development shortcuts
dev-setup: setup
	pip install -e ".[dev]"

format: lint

check-format:
	black --check src/ tests/
	isort --check-only src/ tests/