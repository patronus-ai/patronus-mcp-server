# Patronus MCP Server

A FastMCP server implementation for the Patronus SDK, providing a standardized interface for running powerful evaluations and experiments.

## Features

- **Initialization**: Configure Patronus with project settings and API credentials
- **Evaluation**: Run single evaluations with Patronus Evaluators like Lynx, Glider, and Judge-Image 
- **Experiments**: Run batch experiments with multiple Patronus Evaluators and evaluation datasets

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/patronus-mcp-server.git
cd patronus-mcp-server
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
uv pip install -e .
```

## Usage

### Starting the Server

```bash
python -m src.patronus_mcp.server
```

The server will start with stdio transport, ready to accept commands.

### API Endpoints

#### Initialize
```python
{
    "request": {
        "data": {
            "project_name": "your_project",
            "api_key": "your_api_key",
            "app": "your_app_name"
        }
    }
}
```

#### Evaluate
```python
{
    "request": {
        "data": {
            "task_input": "Your input text",
            "task_context": ["Context 1", "Context 2"],
            "task_output": "Your output text",
            "evaluator": {
                "name": "lynx",
                "criteria": "patronus:hallucination",
                "explain_strategy": "always"
            }
        }
    }
}
```

#### Run Experiment
```python
{
    "request": {
        "data": {
            "project_name": "your_project",
            "experiment_name": "test_experiment",
            "dataset": [
                {
                    "task_input": "Input 1",
                    "task_context": ["Context 1"],
                    "task_output": "Output 1"
                }
            ],
            "evaluators": [
                {
                    "name": "patronus:hallucination",
                    "criteria": "lynx",
                    "explain_strategy": "always"
                }
            ],
            "api_key": "your_api_key"
        }
    }
}
```

### Running Tests

```bash
pytest tests/test_server.py -v
```

## Configuration

The server supports the following configuration options:

- `PATRONUS_API_KEY`: Environment variable for API key
- `project_name`: Project identifier
- `app`: Application name
- `max_concurrency`: Maximum concurrent evaluations (default: 10)

## Development

### Project Structure

```
patronus-mcp-server/
├── src/
│   └── patronus_mcp/
│       └── server.py
├── tests/
│   └── test_server.py
├── pyproject.toml
└── README.md
```

### Adding New Features

1. Define new request models in `server.py`
2. Implement new tool functions with the `@mcp.tool()` decorator
3. Add corresponding tests in `test_server.py`

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
