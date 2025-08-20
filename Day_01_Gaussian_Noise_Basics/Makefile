.PHONY: setup install run demo test lint clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  setup     - Create virtual environment and install dependencies"
	@echo "  install   - Install dependencies only"
	@echo "  run       - Run the noise demo"
	@echo "  demo      - Run complete demonstration pipeline"
	@echo "  test      - Run tests"
	@echo "  lint      - Run code formatting and linting"
	@echo "  clean     - Clean outputs and cache"
	@echo "  help      - Show this help message"

setup:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt

install:
	pip install -r requirements.txt

run:
	python -m src.cli add-noise --config configs/default.yaml --num-batches 2
	python -m src.cli animate --config configs/default.yaml --num-images 16

demo:
	chmod +x scripts/run_noise_demo.sh
	./scripts/run_noise_demo.sh

test:
	pytest tests/ -v

lint:
	black src/ tests/
	isort src/ tests/

clean:
	rm -rf outputs/grids/*
	rm -rf outputs/animations/*
	rm -rf outputs/logs/*
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Quick development targets
noise:
	python -m src.cli add-noise --config configs/default.yaml

animate:
	python -m src.cli animate --config configs/default.yaml

stats:
	python -m src.cli stats --config configs/default.yaml