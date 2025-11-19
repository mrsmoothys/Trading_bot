.PHONY: help test test-unit test-integration test-e2e test-all coverage clean lint format install-dev run-dashboard run-chat

# Default target
help:
	@echo "Trading Bot - Development Makefile"
	@echo "=================================="
	@echo ""
	@echo "Available targets:"
	@echo "  test          - Run all tests with coverage"
	@echo "  test-unit     - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-e2e      - Run end-to-end tests only"
	@echo "  test-fast     - Run fast tests only (skip slow tests)"
	@echo "  coverage      - Generate coverage report"
	@echo "  clean         - Clean up cache and coverage files"
	@echo "  lint          - Run linting (flake8, black check)"
	@echo "  format        - Format code with black"
	@echo "  install-dev   - Install development dependencies"
	@echo "  run-dashboard - Start the dashboard"
	@echo "  run-chat      - Start the chat interface"
	@echo "  backup        - Create system backup"
	@echo "  restore       - Restore from backup"
	@echo ""

# Testing targets
test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/ -v -m unit --tb=short

test-integration:
	pytest tests/ -v -m integration --tb=short

test-e2e:
	pytest tests/ -v -m e2e --tb=short

test-fast:
	pytest tests/ -v -m "not slow" --tb=short

test-all:
	pytest tests/ -v --tb=short --cov=core --cov=ops --cov=data --cov=ui --cov=execution --cov-report=term-missing

# Coverage targets
coverage:
	pytest tests/ --cov=core --cov=ops --cov=data --cov=ui --cov=execution --cov-report=html --cov-report=term
	@echo ""
	@echo "Coverage report generated in htmlcov/"
	@echo "Open htmlcov/index.html to view detailed coverage"

coverage-xml:
	pytest tests/ --cov=core --cov=ops --cov=data --cov=ui --cov=execution --cov-report=xml

# Code quality targets
lint:
	@echo "Running flake8..."
	flake8 core/ ops/ data/ ui/ execution/ tests/ --max-line-length=120 --ignore=E501,W503,E203
	@echo "Running black check..."
	black --check core/ ops/ data/ ui/ execution/ tests/

format:
	@echo "Formatting code with black..."
	black core/ ops/ data/ ui/ execution/ tests/

# Utility targets
clean:
	@echo "Cleaning up..."
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete"

install-dev:
	@echo "Installing development dependencies..."
	pip install pytest pytest-asyncio pytest-cov coverage flake8 black

# Application targets
run-dashboard:
	@echo "Starting dashboard at http://127.0.0.1:8050"
	python -m ui.dashboard

run-chat:
	@echo "Starting chat interface at http://127.0.0.1:8051"
	python -m ui.chat_interface

# Maintenance targets
backup:
	@echo "Creating system backup..."
	python -c "from ops.cache_manager import CacheManager; import asyncio; cm = CacheManager(); asyncio.run(cm.create_backup(compress=True))"

restore:
	@echo "Available backups:"
	@ls -lt backups/*.tar.gz 2>/dev/null | head -10 || echo "No backups found"
	@echo ""
	@echo "To restore, run:"
	@echo "  python -c \"from ops.cache_manager import CacheManager; import asyncio; cm = CacheManager(); asyncio.run(cm.restore_backup('BACKUP_NAME'))\""

# CI/CD targets
ci-test:
	pytest tests/ -v --cov=core --cov=ops --cov=data --cov=ui --cov=execution --cov-report=xml --cov-fail-under=80

ci-lint:
	flake8 core/ ops/ data/ ui/ execution/ tests/ --max-line-length=120 --ignore=E501,W503,E203
