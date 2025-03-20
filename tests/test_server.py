import pytest
import json
import os
from src.patronus_mcp.server import mcp, Request, InitRequest, EvaluationRequest, RemoteEvaluatorConfig, ExperimentRequest

@pytest.fixture
def init_request():
    request = Request(data=InitRequest(
        project_name="MyTest",
        api_key=os.environ.get("PATRONUS_API_KEY"), 
        app="test_app"
    ))
    return {"request": request.model_dump()}

@pytest.fixture
def evaluation_request():
    request = Request(data=EvaluationRequest(
        task_input="What is the capital of France?",
        task_context=["The capital of France is Paris."],
        task_output="Paris is the capital of France.",
        evaluator=RemoteEvaluatorConfig(
            name="lynx",
            criteria="patronus:hallucination",
            explain_strategy="always"
        )
    ))
    return {"request": request.model_dump()}

@pytest.fixture
def experiment_request():
    request = Request(data=ExperimentRequest(
        project_name="MyTest",
        experiment_name="TestExperiment",
        dataset=[{
            "task_input": "What is the capital of France?",
            "task_context": ["The capital of France is Paris."],
            "task_output": "Paris is the capital of France."
        }],
        evaluators=[
            RemoteEvaluatorConfig(
                name="patronus:hallucination",
                criteria="lynx",
                explain_strategy="always"
            )
        ],
        api_key=os.environ.get("PATRONUS_API_KEY"),
    ))
    return {"request": request.model_dump()}

async def test_initialize(init_request):
    response = await mcp.call_tool("initialize", init_request)
    response_data = json.loads(response[0].text)
    assert response_data["status"] == "success"
    assert "Patronus initialized with project: MyTest" in response_data["message"]

async def test_evaluate(evaluation_request):
    response = await mcp.call_tool("evaluate", evaluation_request)
    response_data = json.loads(response[0].text)
    assert response_data["status"] == "success"
    assert "result" in response_data

async def test_run_experiment(experiment_request):
    response = await mcp.call_tool("run_experiment", experiment_request)
    response_data = json.loads(response[0].text)
    assert response_data["status"] == "success"
    assert "results" in response_data