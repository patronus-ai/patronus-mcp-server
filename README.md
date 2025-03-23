# Patronus MCP Server

A FastMCP server implementation for the Patronus SDK, providing a standardized interface for running powerful LLM system optimizations, evaluations, and experiments.

## Features

- Initialize Patronus with API key and project settings
- Run single evaluations with configurable evaluators
- Run batch evaluations with multiple evaluators
- Run experiments with datasets
- Run asynchronous batch evaluations with multiple evaluators

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

3. Install main and dev dependencies:
```bash
uv pip install -e .
uv pip install -e ".[dev]"
```


## Usage

### Initialize

```python
from patronus_mcp.server import mcp, Request, InitRequest

request = Request(data=InitRequest(
    project_name="MyProject",
    api_key="your-api-key",
    app="my-app"
))
response = await mcp.call_tool("initialize", {"request": request.model_dump()})
```

### Single Evaluation

```python
from patronus_mcp.server import Request, EvaluationRequest, RemoteEvaluatorConfig

request = Request(data=EvaluationRequest(
    evaluator=RemoteEvaluatorConfig(
        name="lynx",
        criteria="patronus:hallucination",
        explain_strategy="always"
    ),
    task_input="What is the capital of France?",
    task_output="Paris is the capital of France."
    task_context=["The capital of France is Paris."],
))
response = await mcp.call_tool("evaluate", {"request": request.model_dump()})
```

### Batch Evaluation

```python
from patronus_mcp.server import Request, BatchEvaluationRequest, RemoteEvaluatorConfig

request = Request(data=BatchEvaluationRequest(
    evaluators=[
        RemoteEvaluatorConfig(
            name="lynx",
            criteria="patronus:hallucination",
            explain_strategy="always"
        ),
        RemoteEvaluatorConfig(
            name="judge",
            criteria="patronus:is-concise",
            explain_strategy="always"
        )
    ],
    task_input="What is the capital of France?",
    task_output="Paris is the capital of France."
    task_context=["The capital of France is Paris."],
))
response = await mcp.call_tool("batch_evaluate", {"request": request.model_dump()})
```

### Async Batch Evaluation

```python
from patronus_mcp.server import Request, AsyncBatchEvaluationRequest, AsyncRemoteEvaluatorConfig

request = Request(data=AsyncBatchEvaluationRequest(
    evaluators=[
        AsyncRemoteEvaluatorConfig(
            name="lynx",
            criteria="patronus:hallucination",
            explain_strategy="always"
        ),
        AsyncRemoteEvaluatorConfig(
            name="judge",
            criteria="patronus:is-concise",
            explain_strategy="always"
        )
    ],
    task_input="What is the capital of France?",
    task_output="Paris is the capital of France."
    task_context=["The capital of France is Paris."],
))
response = await mcp.call_tool("async_batch_evaluate", {"request": request.model_dump()})
```

### Run Experiment

```python
from patronus_mcp.server import Request, ExperimentRequest, RemoteEvaluatorConfig

request = Request(data=ExperimentRequest(
    project_name="MyProject",
    experiment_name="MyExperiment",
    dataset=[{
        "task_input": "What is the capital of France?",
        "task_output": "Paris is the capital of France."
        "task_context"=["The capital of France is Paris."],
    }],
    evaluators=[
        RemoteEvaluatorConfig(
            name="lynx",
            criteria="patronus:hallucination",
            explain_strategy="always"
        )
    ]
))
response = await mcp.call_tool("run_experiment", {"request": request.model_dump()})
```

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

### Running Tests

```bash
pytest tests/
```

### Running the Server

```bash
python -m src.patronus_mcp.server
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
