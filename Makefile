.PHONY: test test-verbose test-cov clean install

# Install dependencies
install:
	uv pip install -e .

# Run tests with default output
test:
	pytest

# Run tests with verbose output
test-verbose:
	pytest -v

# Run tests with coverage report
test-cov:
	pytest --cov=src --cov-report=term-missing

# Run tests with print statements visible
test-print:
	pytest -s

# Run a specific test file
test-server:
	pytest tests/test_server.py -v

# Clean up Python cache files
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".coverage" -exec rm -r {} + 