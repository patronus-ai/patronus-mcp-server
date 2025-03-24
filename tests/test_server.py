import pytest
import json
import os
from src.patronus_mcp.server import (
    Request, EvaluationRequest, RemoteEvaluatorConfig, 
    ExperimentRequest, BatchEvaluationRequest, AsyncRemoteEvaluatorConfig, app_factory,
    ListCriteriaRequest
)

@pytest.fixture
def mcp():
    return app_factory(
        patronus_api_key=os.environ.get("PATRONUS_API_KEY"),
        patronus_api_url=os.environ.get("PATRONUS_API_URL", "https://api.patronus.ai")
    )

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
        ]
    ))
    return {"request": request.model_dump()}

@pytest.fixture
def list_evaluators_request():
    request = Request(data=ListEvaluatorsRequest())
    return {"request": request.model_dump()}

async def test_evaluate(mcp, evaluation_request):
    response = await mcp.call_tool("evaluate", evaluation_request)
    response_data = json.loads(response[0].text)
    assert response_data["status"] == "success"
    assert "result" in response_data

async def test_run_experiment(mcp, experiment_request):
    response = await mcp.call_tool("run_experiment", experiment_request)
    response_data = json.loads(response[0].text)
    assert response_data["status"] == "success"
    assert "results" in response_data

async def test_batch_evaluate(mcp, batch_evaluation_request):
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

async def test_batch_evaluate_error(mcp):
    invalid_request = Request(data=BatchEvaluationRequest(
        task_input="What is the capital of France?",
        task_output="Paris is the capital of France.",
        evaluators=[
            AsyncRemoteEvaluatorConfig(
                name="invalid_evaluator",  # This should cause an error
                criteria="invalid_criteria"
            )
        ]
    ))
    
    response = await mcp.call_tool("batch_evaluate", {"request": invalid_request.model_dump()})
    response_data = json.loads(response[0].text)
    assert response_data["status"] == "error"
    assert "message" in response_data

async def test_batch_evaluate_empty(mcp):
    empty_request = Request(data=BatchEvaluationRequest(
        task_input="What is the capital of France?",
        task_output="Paris is the capital of France.",
        evaluators=[]  
    ))
    
    response = await mcp.call_tool("batch_evaluate", {"request": empty_request.model_dump()})
    response_data = json.loads(response[0].text)
    assert response_data["status"] == "error"
    assert "message" in response_data

async def test_list_evaluators(mcp):
    """Test listing evaluators"""
    response = await mcp.call_tool("list_evaluators", {})
    response_data = json.loads(response[0].text)
    assert response_data["status"] == "success"
    assert "result" in response_data
    result = response_data["result"]
    
    # Verify the response structure
    assert isinstance(result, list)
    assert len(result) > 0
    
    # Verify the structure of each evaluator
    evaluator = result[0]
    assert isinstance(evaluator, dict)
    assert "id" in evaluator
    assert "name" in evaluator
    assert "evaluator_family" in evaluator
    assert "aliases" in evaluator
    assert isinstance(evaluator["aliases"], list)
    
    # Verify specific evaluator types are present
    evaluator_ids = [e["id"] for e in result]
    assert any("judge" in id for id in evaluator_ids)
    assert any("hallucination" in id for id in evaluator_ids)
    assert any("toxicity" in id for id in evaluator_ids)

async def test_list_evaluators_no_client():
    """Test listing evaluators without client"""
    # Create an MCP instance without a client
    mcp_no_client = app_factory(
        patronus_api_key=None,
        patronus_api_url="https://api.patronus.ai"
    )
    
    response = await mcp_no_client.call_tool("list_evaluators", {})
    response_data = json.loads(response[0].text)
    assert response_data["status"] == "error"

async def test_list_criteria(mcp):
    """Test listing evaluators"""
    response = await mcp.call_tool("list_criteria", {})
    response_data = json.loads(response[0].text)
    print(response_data)
    assert response_data["status"] == "success"
    assert "result" in response_data
    result = response_data["result"]
    
    # Verify the response structure
    assert isinstance(result, list)
    assert len(result) > 0
    
    # Verify the structure of each evaluator
    criterion = result[0]
    assert "public_id" in criterion
    assert "evaluator_family" in criterion
    assert "name" in criterion
    assert "revision" in criterion
    assert "config" in criterion
    assert "is_patronus_managed" in criterion
    assert "created_at" in criterion
    assert "description" in criterion
    assert isinstance(criterion["config"], dict)
    
    # Verify specific criterion types are present
    criteria_names = [e["name"] for e in result]
    assert any("patronus:hallucination" in id for id in criteria_names)
    assert any("patronus:is-concise" in id for id in criteria_names)
    assert any("patronus:caption-describes-non-primary-objects" in id for id in criteria_names)

async def test_list_criteria_no_client():
    """Test list_criteria when no client is provided"""
    # Create an MCP instance without a client
    mcp_no_client = app_factory(
        patronus_api_key=None,
        patronus_api_url="https://api.patronus.ai"
    )
    response = await mcp_no_client.call_tool("list_criteria", {})
    response_data = json.loads(response[0].text)
    assert response_data["status"] == "error"
