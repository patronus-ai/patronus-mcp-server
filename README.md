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

### Running the Server

The server can be run with an API key provided in two ways:

1. Command line argument:
```bash
python src/patronus_mcp/server.py --api-key your_api_key_here
```

2. Environment variable:
```bash
export PATRONUS_API_KEY=your_api_key_here
python src/patronus_mcp/server.py
```

### Interactive Testing

The test script (`tests/test_live.py`) provides an interactive way to test different evaluation endpoints. You can run it in several ways:

1. With API key in command line:
```bash
python -m tests.test_live src/patronus_mcp/server.py --api-key your_api_key_here
```

2. With API key in environment:
```bash
export PATRONUS_API_KEY=your_api_key_here
python -m tests.test_live src/patronus_mcp/server.py
```

3. Without API key (will prompt):
```bash
python -m tests.test_live src/patronus_mcp/server.py
```

The test script provides three test options:
1. Single evaluation test
2. Batch evaluation test
3. Async batch evaluation test

Each test will display the results in a nicely formatted JSON output.

### API Usage

#### Initialize

```python
from patronus_mcp.server import mcp, Request, InitRequest

request = Request(data=InitRequest(
    project_name="MyProject",
    api_key="your-api-key",
    app="my-app"
))
response = await mcp.call_tool("initialize", {"request": request.model_dump()})
```

#### Single Evaluation

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

#### Batch Evaluation

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

#### Async Batch Evaluation

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

#### Run Experiment

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
│       ├── __init__.py
│       └── server.py
├── tests/
│   └── test_server.py
    └── test_live.py
├── pyproject.toml
└── README.md
```

### Adding New Features

1. Define new request models in `server.py`:
   ```python
   class NewFeatureRequest(BaseModel):
       # Define your request fields here
       field1: str
       field2: Optional[int] = None
   ```

2. Implement new tool functions with the `@mcp.tool()` decorator:
   ```python
   @mcp.tool()
   def new_feature(request: Request[NewFeatureRequest]):
       # Implement your feature logic here
       return {"status": "success", "result": ...}
   ```

3. Add corresponding tests:
   - Add API tests in `test_server.py`:
     ```python
     def test_new_feature():
         request = Request(data=NewFeatureRequest(
             field1="test",
             field2=123
         ))
         response = mcp.call_tool("new_feature", {"request": request.model_dump()})
         assert response["status"] == "success"
     ```
   - Add interactive test in `test_live.py`:
     ```python
     async def test_new_feature(self):
         request = Request(data=NewFeatureRequest(
             field1="test",
             field2=123
         ))
         result = await self.session.call_tool("new_feature", {"request": request.model_dump()})
         await self._handle_response(result, "New feature test")
     ```
   - Add the new test to the test selection menu in `main()`

4. Update the README with:
   - New feature description in the Features section
   - API usage example in the API Usage section
   - Any new configuration options or requirements

### Running Tests

The test script uses the Model Context Protocol (MCP) client to communicate with the server. It supports:
- Interactive test selection
- JSON response formatting
- Proper resource cleanup
- Multiple API key input methods

You can also run the standard test suite:
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
