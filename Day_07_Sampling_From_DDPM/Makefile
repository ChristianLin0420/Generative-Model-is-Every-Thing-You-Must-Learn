.PHONY: setup sample compare test clean

# Setup environment
setup:
	pip install -r requirements.txt

# Generate sample grids
sample:
	python -m src.cli sample.grid --config configs/mnist.yaml
	python -m src.cli sample.grid --config configs/cifar10.yaml

# Generate sample trajectories
sample-traj:
	python -m src.cli sample.traj --config configs/mnist.yaml --idx 0
	python -m src.cli sample.traj --config configs/cifar10.yaml --idx 0

# Compare checkpoints
compare:
	python -m src.cli compare.ckpts --config configs/mnist.yaml
	python -m src.cli compare.ckpts --config configs/cifar10.yaml

# Evaluate quality metrics
eval:
	python -m src.cli eval.quality --config configs/mnist.yaml
	python -m src.cli eval.quality --config configs/cifar10.yaml

# Run tests
test:
	pytest -v tests/

# Quick test
test-quick:
	pytest -q

# Clean outputs
clean:
	rm -rf outputs/grids/*
	rm -rf outputs/animations/*
	rm -rf outputs/curves/*
	rm -rf outputs/logs/*
	rm -rf outputs/reports/*

# Create output directories
create-dirs:
	mkdir -p outputs/grids
	mkdir -p outputs/animations
	mkdir -p outputs/curves
	mkdir -p outputs/logs
	mkdir -p outputs/reports
