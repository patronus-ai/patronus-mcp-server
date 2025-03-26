# Patronus MCP Server

An MCP server implementation for the Patronus SDK, providing a standardized interface for running powerful LLM system optimizations, evaluations, and experiments.

## Features

- Initialize Patronus with API key and project settings
- Run single evaluations with configurable evaluators
- Run batch evaluations with multiple evaluators
- Run experiments with datasets

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
response = await mcp.call_tool("batch_evaluate", {"request": request.model_dump()})
```

#### Run Experiment

```python
from patronus_mcp import Request, ExperimentRequest, RemoteEvaluatorConfig, CustomEvaluatorConfig

# Create a custom evaluator function
@evaluator()
def exact_match(expected: str, actual: str, case_sensitive: bool = False) -> bool:
    if not case_sensitive:
        return expected.lower() == actual.lower()
    return expected == actual

# Create a custom adapter class
class ExactMatchAdapter(FuncEvaluatorAdapter):
    def __init__(self, case_sensitive: bool = False):
        super().__init__(exact_match)
        self.case_sensitive = case_sensitive

    def transform(self, row, task_result, parent, **kwargs):
        args = []
        evaluator_kwargs = {
            "expected": row.gold_answer,
            "actual": task_result.output if task_result else "",
            "case_sensitive": self.case_sensitive
        }
        return args, evaluator_kwargs

# Create experiment request
request = Request(data=ExperimentRequest(
    project_name="my_project",
    experiment_name="my_experiment",
    dataset=[{
        "input": "What is 2+2?",
        "output": "4",
        "gold_answer": "4"
    }],
    evaluators=[
        # Remote evaluator
        RemoteEvaluatorConfig(
            name="judge",
            criteria="patronus:is-concise"
        ),
        # Custom evaluator
        CustomEvaluatorConfig(
            adapter_class="my_module.ExactMatchAdapter",
            adapter_kwargs={"case_sensitive": False}
        )
    ]
))

# Run the experiment
response = await mcp.call_tool("run_experiment", {"request": request.model_dump()})
response_data = json.loads(response[0].text)

# The experiment runs asynchronously, so results will be pending initially
assert response_data["status"] == "success"
assert "results" in response_data
assert isinstance(response_data["results"], str)  # Results will be a string (pending)
```

#### List Evaluator Info

Get a comprehensive view of all available evaluators and their associated criteria:

```python
# No request body needed
response = await mcp.call_tool("list_evaluator_info", {})

# Response structure:
{
    "status": "success",
    "result": {
        "evaluator_family_name": {
            "evaluator": {
                # evaluator configuration and metadata
            },
            "criteria": [
                # list of available criteria for this evaluator
            ]
        }
    }
}
```

This endpoint combines information about evaluators and their associated criteria into a single, organized response. The results are grouped by evaluator family, with each family containing its evaluator configuration and a list of available criteria.

#### Create Criteria

Creates a new evaluator criteria in the Patronus API.

```python
{
    "request": {
        "data": {
            "name": "my-criteria",
            "evaluator_family": "Judge",
            "config": {
                "pass_criteria": "The MODEL_OUTPUT should contain all the details needed from RETRIEVED CONTEXT to answer USER INPUT.",
                "active_learning_enabled": false,
                "active_learning_negative_samples": null,
                "active_learning_positive_samples": null
            }
        }
    }
}
```

Parameters:
- `name` (str): Unique name for the criteria
- `evaluator_family` (str): Family of the evaluator (e.g., "Judge", "Answer Relevance")
- `config` (dict): Configuration for the criteria
  - `pass_criteria` (str): The criteria that must be met for a pass
  - `active_learning_enabled` (bool, optional): Whether active learning is enabled
  - `active_learning_negative_samples` (int, optional): Number of negative samples for active learning
  - `active_learning_positive_samples` (int, optional): Number of positive samples for active learning

Returns:
```python
{
    "status": "success",
    "result": {
        "name": "my-criteria",
        "evaluator_family": "Judge",
        "config": {
            "pass_criteria": "The MODEL_OUTPUT should contain all the details needed from RETRIEVED CONTEXT to answer USER INPUT.",
            "active_learning_enabled": False,
            "active_learning_negative_samples": null,
            "active_learning_positive_samples": null
        }
    }
}
```

#### Custom Evaluate

Evaluates a task output using a custom evaluator function decorated with `@evaluator`.

```python
{
    "request": {
        "data": {
            "task_input": "What is the capital of France?",
            "task_context": ["The capital of France is Paris."],
            "task_output": "Paris is the capital of France.",
            "evaluator_function": "is_concise",
            "evaluator_args": {
                "threshold": 0.7
            }
        }
    }
}
```

Parameters:
- `task_input` (str): The input prompt
- `task_context` (List[str], optional): Context information for the evaluation
- `task_output` (str): The output to evaluate
- `evaluator_function` (str): Name of the evaluator function to use (must be decorated with `@evaluator`)
- `evaluator_args` (Dict[str, Any], optional): Additional arguments for the evaluator function

The evaluator function can return:
- `bool`: Simple pass/fail result
- `int` or `float`: Numeric score (pass threshold is 0.7)
- `str`: Text output
- `EvaluationResult`: Full evaluation result with score, pass status, explanation, etc.

Returns:
```python
{
    "status": "success",
    "result": {
        "score": 0.8,
        "pass_": true,
        "text_output": "Good match",
        "explanation": "Output matches context well",
        "metadata": {
            "context_length": 1
        },
        "tags": ["high_score"]
    }
}
```

Example evaluator function:
```python
from patronus import evaluator, EvaluationResult

@evaluator
def is_concise(output: str) -> bool:
    """Simple evaluator that checks if the output is concise"""
    return len(output.split()) < 10

@evaluator
def has_score(output: str, context: List[str]) -> EvaluationResult:
    """Evaluator that returns a score based on context"""
    return EvaluationResult(
        score=0.8,
        pass_=True,
        text_output="Good match",
        explanation="Output matches context well",
        metadata={"context_length": len(context)},
        tags=["high_score"]
    )
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
