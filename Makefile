# Makefile for Quantum Trader Pro
# Common development tasks

.PHONY: help install install-dev test lint format type-check security clean setup-hooks check-deps

# Default target
help:
	@echo "Quantum Trader Pro - Development Commands"
	@echo "========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make setup-hooks    Setup pre-commit hooks"
	@echo ""
	@echo "Quality:"
	@echo "  make lint           Run linters (flake8)"
	@echo "  make format         Format code (black + isort)"
	@echo "  make type-check     Run type checking (mypy)"
	@echo "  make security       Run security checks (bandit)"
	@echo "  make check          Run all quality checks"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-cov       Run tests with coverage"
	@echo ""
	@echo "Other:"
	@echo "  make check-deps     Check dependencies"
	@echo "  make clean          Clean generated files"
	@echo "  make run            Run the trading bot"

# Installation
install:
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt

setup-hooks:
	pip install pre-commit
	pre-commit install

# Code quality
lint:
	@echo "Running flake8..."
	flake8 --config=.flake8 .
	@echo "✓ Linting passed"

format:
	@echo "Formatting with black..."
	black --line-length 100 .
	@echo "Sorting imports with isort..."
	isort --profile black --line-length 100 .
	@echo "✓ Formatting complete"

type-check:
	@echo "Running mypy..."
	mypy --ignore-missing-imports --no-strict-optional .
	@echo "✓ Type checking passed"

security:
	@echo "Running bandit security checks..."
	bandit -r . -ll --skip B101,B311 --exclude tests/
	@echo "✓ Security checks passed"

check: lint type-check security
	@echo "✓ All quality checks passed"

# Testing
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v -m unit

test-cov:
	pytest tests/ --cov=. --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/"

# Dependencies
check-deps:
	python check_dependencies.py

# Cleaning
clean:
	@echo "Cleaning generated files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true
	@echo "✓ Clean complete"

# Run
run:
	python main.py

# Development workflow
dev-check: format lint test-unit
	@echo "✓ Development checks passed - ready to commit"
