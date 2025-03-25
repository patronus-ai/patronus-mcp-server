import pytest
import json
import os
from src.patronus_mcp.server import (
    Request, EvaluationRequest, RemoteEvaluatorConfig, 
    ExperimentRequest, BatchEvaluationRequest, AsyncRemoteEvaluatorConfig, app_factory,
    CreateCriteriaRequest
)
from patronus import evaluator, EvaluationResult
from typing import List

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
def create_criteria_request():
    """Fixture for create criteria request"""
    request = Request(data=CreateCriteriaRequest(
        name="answer-incompleteness-test-1",
        evaluator_family="Judge",
        config={
            "pass_criteria": "The MODEL_OUTPUT should contain all the details needed from RETRIEVED CONTEXT to answer USER INPUT.",
            "active_learning_enabled": False,
            "active_learning_negative_samples": None,
            "active_learning_positive_samples": None
        }
    ))
    return {"request": request.model_dump()}

@pytest.fixture
def custom_evaluation_request():
    """Fixture for custom evaluation request"""
    request = Request(data={
        "evaluator_function": is_concise,
        "args": ["Paris is the capital of France."]
    })
    return {"request": request.model_dump()}

@pytest.fixture
def custom_evaluation_request_with_args():
    """Fixture for custom evaluation request with additional arguments"""
    request = Request(data={
        "evaluator_function": has_score,
        "args": ["Paris is the capital of France.", ["The capital of France is Paris."]]
    })
    return {"request": request.model_dump()}

@evaluator()
def is_concise(output: str) -> bool:
    """Simple evaluator that checks if the output is concise"""
    return len(output.split()) < 10

@evaluator()
def has_score(output: str, context: List[str]) -> EvaluationResult:
    """Evaluator that returns a score based on context"""
    return EvaluationResult(
        score=0.8,
        pass_=True,
        text_output="Good match",
        explanation="Output matches context well",
        metadata={"context_length": len(context)},
        tags={"quality": "high_score"}
    )

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

async def test_list_evaluator_info(mcp):
    """Test list_evaluator_info returns combined evaluator and criteria information"""
    # Call the tool
    response = await mcp.call_tool("list_evaluator_info", {})
    response_data = json.loads(response[0].text)
    print("response_data", response_data)
    # Check basic response structure
    assert response_data["status"] == "success"
    assert isinstance(response_data["result"], dict)
    
    # Check that at least one evaluator family exists
    assert len(response_data["result"]) > 0
    
    # Check structure for first evaluator family
    first_family = next(iter(response_data["result"]))
    family_data = response_data["result"][first_family]
    
    # Verify the structure of the response
    assert "evaluator" in family_data
    assert "criteria" in family_data
    assert isinstance(family_data["criteria"], list)
    
    # Verify evaluator does not contain removed fields
    evaluator = family_data["evaluator"]
    assert "evaluator_family" not in evaluator
    assert "revision" not in evaluator
    assert "name" not in evaluator
    
    # If criteria exist, verify they don't contain removed fields
    if family_data["criteria"]:
        criterion = family_data["criteria"][0]
        assert "evaluator_family" not in criterion
        assert "revision" not in criterion

async def test_list_evaluator_info_no_client():
    """Test list_evaluator_info when no client is provided"""
    # Create MCP instance without client
    mcp_no_client = app_factory(
        patronus_api_key=None,
        patronus_api_url="https://api.patronus.ai"
    )
    response = await mcp_no_client.call_tool("list_evaluator_info", {})
    response_data = json.loads(response[0].text)
    assert response_data["status"] == "error"

async def test_create_criteria(mcp, create_criteria_request):
    """Test creating a new criteria"""
    print("create_criteria_request", create_criteria_request)
    response = await mcp.call_tool("create_criteria", create_criteria_request)
    response_data = json.loads(response[0].text)
    print("response_data", response_data)

    assert response_data["status"] == "success"
    assert "result" in response_data

async def test_create_criteria_no_client(create_criteria_request):
    """Test create_criteria when no client is provided"""
    mcp_no_client = app_factory(
        patronus_api_key=None,
        patronus_api_url="https://api.patronus.ai"
    )
    
    response = await mcp_no_client.call_tool("create_criteria", create_criteria_request)
    response_data = json.loads(response[0].text)
    assert response_data["status"] == "error"

async def test_custom_evaluate(mcp, custom_evaluation_request):
    """Test custom evaluation with a simple boolean evaluator"""
    response = await mcp.call_tool("custom_evaluate", custom_evaluation_request)
    response_data = json.loads(response[0].text)
    assert response_data["status"] == "success"
    assert "result" in response_data
    result = response_data["result"]
    assert "score" in result
    assert "pass_" in result
    assert "text_output" in result
    assert "explanation" in result
    assert "metadata" in result
    assert "tags" in result

async def test_custom_evaluate_with_args(mcp, custom_evaluation_request_with_args):
    """Test custom evaluation with an evaluator that returns EvaluationResult"""
    response = await mcp.call_tool("custom_evaluate", custom_evaluation_request_with_args)
    response_data = json.loads(response[0].text)
    assert response_data["status"] == "success"
    assert "result" in response_data
    result = response_data["result"]
    assert result["score"] == 0.8
    assert result["pass_"] is True
    assert result["text_output"] == "Good match"
    assert result["explanation"] == "Output matches context well"
    assert result["metadata"]["context_length"] == 1
    assert "quality" in result["tags"]

async def test_custom_evaluate_invalid_function(mcp):
    """Test custom evaluation with an invalid evaluator function"""
    request = Request(data={
        "evaluator_function": "invalid_function",
        "args": ["Paris is the capital of France."]
    })
    response = await mcp.call_tool("custom_evaluate", {"request": request.model_dump()})
    response_data = json.loads(response[0].text)
    assert response_data["status"] == "error"
