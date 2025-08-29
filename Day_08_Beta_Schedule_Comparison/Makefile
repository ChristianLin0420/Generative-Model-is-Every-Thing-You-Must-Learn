.PHONY: setup train_all sample_all compare test clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  setup      - Install dependencies and setup environment"
	@echo "  train_all  - Train models with all beta schedules (linear, cosine, quadratic)"
	@echo "  sample_all - Generate samples and animations for all schedules"
	@echo "  compare    - Compare metrics across schedules and generate report"
	@echo "  test       - Run all tests"
	@echo "  clean      - Clean outputs and temporary files"

# Setup environment
setup:
	pip install -e .
	pip install -r requirements.txt

# Train all schedules
train_all:
	bash scripts/train_all.sh

# Generate samples for all schedules
sample_all:
	bash scripts/sample_all.sh

# Compare schedules and generate report
compare:
	bash scripts/compare.sh

# Run tests
test:
	python -m pytest tests/ -v

# Clean outputs
clean:
	rm -rf outputs/*/ckpts/*
	rm -rf outputs/*/logs/*
	rm -rf outputs/*/curves/*
	rm -rf outputs/*/grids/*
	rm -rf outputs/*/animations/*
	rm -rf outputs/plots/*
	rm -rf outputs/comparison/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
