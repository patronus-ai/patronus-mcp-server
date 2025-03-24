.PHONY: test test-verbose test-cov clean install run-server run-server-custom test-live

# Install dependencies
install:
	uv pip install -e .

# Install dev dependencies
install-dev:
	uv pip install -e ".[dev]"

# Run tests with default output
test:
	pytest tests/

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

# Run the server (uses PATRONUS_API_KEY from env if not provided)
run-server:
	@if [ -z "$$PATRONUS_API_KEY" ]; then \
		echo "Error: PATRONUS_API_KEY environment variable is not set"; \
		echo "Please set it with: export PATRONUS_API_KEY=your_key_here"; \
		exit 1; \
	fi
	python -m src.patronus_mcp.server

# Run the server with custom URL (uses PATRONUS_API_KEY and PATRONUS_API_URL from env if not provided)
run-server-custom:
	@if [ -n "$(API_KEY)" ] && [ -n "$(API_URL)" ]; then \
		python -m src.patronus_mcp.server --api-key $(API_KEY) --api-url $(API_URL); \
	elif [ -n "$(API_KEY)" ]; then \
		python -m src.patronus_mcp.server --api-key $(API_KEY); \
	elif [ -n "$(API_URL)" ]; then \
		python -m src.patronus_mcp.server --api-url $(API_URL); \
	else \
		python -m src.patronus_mcp.server; \
	fi

# Run live test script
test-live:
	@if [ -z "$$PATRONUS_API_KEY" ]; then \
		echo "Error: PATRONUS_API_KEY environment variable is not set"; \
		echo "Please set it with: export PATRONUS_API_KEY=your_key_here"; \
		exit 1; \
	fi
	python -m tests.test_live src/patronus_mcp/server.py

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