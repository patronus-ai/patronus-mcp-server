import pytest
import json
import os
from src.patronus_mcp.server import mcp, Request, InitRequest, EvaluationRequest, RemoteEvaluatorConfig, ExperimentRequest, BatchEvaluationRequest, AsyncBatchEvaluationRequest, AsyncRemoteEvaluatorConfig

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

@pytest.fixture
def batch_evaluation_request():
    request = Request(data=BatchEvaluationRequest(
        task_input="What is the capital of France?",
        task_context=["The capital of France is Paris."],
        task_output="Paris is the capital of France.",
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
        ]
    ))
    return {"request": request.model_dump()}

@pytest.fixture
def async_batch_evaluation_request():
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
        task_output="Paris is the capital of France.",
        system_prompt="You are a helpful assistant.",
        task_context=["The capital of France is Paris."],
        task_metadata={"source": "test"}
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

async def test_batch_evaluate(batch_evaluation_request):
    response = await mcp.call_tool("batch_evaluate", batch_evaluation_request)
    response_data = json.loads(response[0].text)
    assert response_data["status"] == "success"
    assert "results" in response_data
    results = response_data["results"]
    
    assert "all_succeeded" in results
    assert "failed_evaluations" in results
    assert "succeeded_evaluations" in results

    for eval_result in results["succeeded_evaluations"] + results["failed_evaluations"]:
        assert "score" in eval_result
        assert "pass_" in eval_result
        assert "text_output" in eval_result
        assert "metadata" in eval_result
        assert "explanation" in eval_result
        assert "tags" in eval_result
        assert "dataset_id" in eval_result
        assert "dataset_sample_id" in eval_result
        assert "evaluation_duration" in eval_result
        assert "explanation_duration" in eval_result

async def test_batch_evaluate_error():
    invalid_request = Request(data=BatchEvaluationRequest(
        task_input="What is the capital of France?",
        task_output="Paris is the capital of France.",
        evaluators=[
            RemoteEvaluatorConfig(
                name="invalid_evaluator",  # This should cause an error
                criteria="invalid_criteria"
            )
        ]
    ))
    
    response = await mcp.call_tool("batch_evaluate", {"request": invalid_request.model_dump()})
    response_data = json.loads(response[0].text)
    assert response_data["status"] == "error"
    assert "message" in response_data


async def test_batch_evaluate_empty():
    empty_request = Request(data=BatchEvaluationRequest(
        task_input="What is the capital of France?",
        task_output="Paris is the capital of France.",
        evaluators=[]  
    ))
    
    response = await mcp.call_tool("batch_evaluate", {"request": empty_request.model_dump()})
    response_data = json.loads(response[0].text)
    assert response_data["status"] == "error"
    assert "message" in response_data

async def test_async_batch_evaluate(async_batch_evaluation_request):
    response = await mcp.call_tool("async_batch_evaluate", async_batch_evaluation_request)
    response_data = json.loads(response[0].text)
    print(response_data)
    assert response_data["status"] == "success"
    assert "results" in response_data
    results = response_data["results"]
    
    assert "all_succeeded" in results
    assert "failed_evaluations" in results
    assert "succeeded_evaluations" in results

    for eval_result in results["succeeded_evaluations"] + results["failed_evaluations"]:
        assert "score" in eval_result
        assert "pass_" in eval_result
        assert "text_output" in eval_result
        assert "metadata" in eval_result
        assert "explanation" in eval_result
        assert "tags" in eval_result
        assert "dataset_id" in eval_result
        assert "dataset_sample_id" in eval_result
        assert "evaluation_duration" in eval_result
        assert "explanation_duration" in eval_result

async def test_async_batch_evaluate_error():
    invalid_request = Request(data=AsyncBatchEvaluationRequest(
        task_input="What is the capital of France?",
        task_output="Paris is the capital of France.",
        evaluators=[
            AsyncRemoteEvaluatorConfig(
                name="invalid_evaluator",  # This should cause an error
                criteria="invalid_criteria"
            )
        ]
    ))
    
    response = await mcp.call_tool("async_batch_evaluate", {"request": invalid_request.model_dump()})
    response_data = json.loads(response[0].text)
    assert response_data["status"] == "error"
    assert "message" in response_data

async def test_async_batch_evaluate_empty():
    empty_request = Request(data=AsyncBatchEvaluationRequest(
        task_input="What is the capital of France?",
        task_output="Paris is the capital of France.",
        evaluators=[]
    ))
    
    response = await mcp.call_tool("async_batch_evaluate", {"request": empty_request.model_dump()})
    response_data = json.loads(response[0].text)
    assert response_data["status"] == "error"
    assert "message" in response_data