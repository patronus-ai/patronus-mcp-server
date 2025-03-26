import argparse
import os 
import httpx
import json

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
import patronus, patronus.evals, patronus.experiments.experiment 
from typing import Optional, List, Dict, Any, Literal, Generic, TypeVar, Union
from patronus.api.api_client import PatronusAPIClient
from patronus.api.api_types import ListEvaluatorsResponse, ListCriteriaResponse
from patronus import evaluator, EvaluationResult

T = TypeVar('T')

class Request(BaseModel, Generic[T]):
    data: T

class InitRequest(BaseModel):
    project_name: Optional[str] = None
    app: Optional[str] = None
    api_url: Optional[str] = None
    otel_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    service: Optional[str] = None

class RemoteEvaluatorConfig(BaseModel):
    name: str  
    criteria: Optional[str] = None  
    explain_strategy: Optional[Literal["never", "on-fail", "on-success", "always"]] = "always"
    criteria_config: Optional[Dict[str, Any]] = None  
    allow_update: Optional[bool] = False
    max_attempts: Optional[int] = 3

class AsyncRemoteEvaluatorConfig(BaseModel):
    name: str  
    criteria: Optional[str] = None  
    explain_strategy: Optional[Literal["never", "on-fail", "on-success", "always"]] = "always"
    criteria_config: Optional[Dict[str, Any]] = None  
    allow_update: Optional[bool] = False
    max_attempts: Optional[int] = 3

class EvaluationRequest(BaseModel):
    evaluator: RemoteEvaluatorConfig
    system_prompt: Optional[str] = None
    task_context: Union[list[str], str, None] = None
    task_attachments: Union[list[Any], None] = None
    task_input: Optional[str] = None
    task_output: Optional[str] = None
    gold_answer: Optional[str] = None
    task_metadata: Optional[Dict[str, Any]] = None

class ExperimentRequest(BaseModel):
    project_name: str
    experiment_name: str
    dataset: List[Dict[str, Any]]  
    evaluators: List[RemoteEvaluatorConfig]
    tags: Optional[Dict[str, str]] = None
    max_concurrency: int = 10
    api_key: Optional[str] = None

class BatchEvaluationRequest(BaseModel):
    evaluators: List[AsyncRemoteEvaluatorConfig]
    task_input: Optional[str] = None
    task_output: Optional[str] = None
    system_prompt: Optional[str] = None
    task_context: Union[list[str], str, None] = None
    task_attachments: Union[list[Any], None] = None
    gold_answer: Optional[str] = None
    task_metadata: Optional[Dict[str, Any]] = None

class ListCriteriaRequest(BaseModel):
    """Request model for listing criteria"""
    evaluator_family: Optional[str] = None
    evaluator_id: Optional[str] = None
    get_last_revision: bool = False
    is_patronus_managed: Optional[bool] = None
    limit: int = 1000
    name: Optional[str] = None
    offset: int = 0
    public_id: Optional[str] = None
    revision: Optional[str] = None

class CreateCriteriaRequest(BaseModel):
    name: str
    evaluator_family: str
    config: Dict[str, Any]

def _create_evaluator(config: RemoteEvaluatorConfig) -> Any:
    kwargs = {}
    if config.criteria is not None:
        kwargs['criteria'] = config.criteria
    if config.explain_strategy is not None:
        kwargs['explain_strategy'] = config.explain_strategy
    if config.criteria_config is not None:
        kwargs['criteria_config'] = config.criteria_config
    if config.allow_update is not None:
        kwargs['allow_update'] = config.allow_update
    if config.max_attempts is not None:
        kwargs['max_attempts'] = config.max_attempts
        
    return patronus.RemoteEvaluator(config.name, **kwargs)

def evaluate(request: Request[EvaluationRequest]):
    try:
        evaluator = _create_evaluator(request.data.evaluator)
        eval_kwargs = {}
        
        fields = {
            "task_input",
            "task_output",
            "system_prompt",
            "task_context",
            "task_attachments",
            "gold_answer",
            "task_metadata"
        }
        
        eval_kwargs.update({
            field: getattr(request.data, field)
            for field in fields
            if getattr(request.data, field) is not None
        })
        
        result = evaluator.evaluate(**eval_kwargs)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def run_experiment(request: Request[ExperimentRequest]):
    try:
        evaluators = [
            _create_evaluator(config)
            for config in request.data.evaluators
        ]
        
        fields = {
            "project_name",
            "experiment_name",
            "dataset",
            "tags",
            "max_concurrency",
            "api_key"
        }
        
        experiment_kwargs = {
            field: getattr(request.data, field)
            for field in fields
            if getattr(request.data, field) is not None
        }
        
        result = patronus.experiments.experiment.run_experiment(
            evaluators=evaluators,
            **experiment_kwargs
        )
        
        if hasattr(result, 'to_dict'):
            result_dict = result.to_dict()
        else:
            result_dict = str(result)
            
        return {"status": "success", "results": result_dict}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def _create_async_evaluator(config: AsyncRemoteEvaluatorConfig) -> Any:
    kwargs = {}
    if config.criteria is not None:
        kwargs['criteria'] = config.criteria
    if config.explain_strategy is not None:
        kwargs['explain_strategy'] = config.explain_strategy
    if config.criteria_config is not None:
        kwargs['criteria_config'] = config.criteria_config
    if config.allow_update is not None:
        kwargs['allow_update'] = config.allow_update
    if config.max_attempts is not None:
        kwargs['max_attempts'] = config.max_attempts
        
    return patronus.evals.AsyncRemoteEvaluator(config.name, **kwargs)

async def batch_evaluate(request: Request[BatchEvaluationRequest]):
    try:
        if not request.data.evaluators:
            return {"status": "error", "message": "No evaluators provided"}
            
        evaluators = [
            _create_async_evaluator(config)
            for config in request.data.evaluators
        ]
        
        eval_kwargs = {}
        fields = {
            "task_input",
            "task_output",
            "system_prompt",
            "task_context",
            "task_attachments",
            "gold_answer",
            "task_metadata"
        }
        
        eval_kwargs.update({
            field: getattr(request.data, field)
            for field in fields
            if getattr(request.data, field) is not None
        })
        
        async with patronus.AsyncPatronus() as client:
            results = await client.evaluate(
                evaluators=evaluators,
                **eval_kwargs
            )
            
            # Convert results to a serializable format
            results_dict = {
                "all_succeeded": results.all_succeeded(),
                "failed_evaluations": [
                    eval.model_dump(mode="json")
                    for eval in results.failed_evaluations()
                ],
                "succeeded_evaluations": [
                    eval.model_dump(mode="json")
                    for eval in results.succeeded_evaluations()
                ]
            }
            
            return {"status": "success", "results": results_dict}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

def _list_evaluators(patronus_client: PatronusAPIClient = None) -> List[Dict[str, Any]]:
    """List all available evaluators from the Patronus API"""
    if not patronus_client:
        raise ValueError("Patronus client must be provided")
    
    response = patronus_client.list_evaluators_sync()
    return [evaluator.model_dump() for evaluator in response]

def _list_criteria(request: Request[ListCriteriaRequest], patronus_client: PatronusAPIClient = None) -> List[Dict[str, Any]]:
    """List all available criteria from the Patronus API"""
    if not patronus_client:
        raise ValueError("Patronus client must be provided")
    
    criteria_request = request.data if request.data else ListCriteriaRequest()
    response = patronus_client.list_criteria_sync(request=criteria_request)
    return [criterion.model_dump() for criterion in response.evaluator_criteria]

def create_criteria(request: Request[CreateCriteriaRequest], patronus_client: PatronusAPIClient = None) -> Dict[str, Any]:
    """
    Create a new criteria using the Patronus API.
    Args:
        request: Dictionary containing the request data with structure:
            {
                "request": {
                    "data": {
                        "name": str,
                        "evaluator_family": str,
                        "config": Dict[str, Any]
                    }
                }
            }
        patronus_client: Patronus API client instance
    Returns:
        Dictionary containing the created criteria information
    """
    try:
        if not patronus_client:
            raise ValueError("Patronus client must be provided")
        request_obj = CreateCriteriaRequest(**request["data"])
        response = patronus_client.create_criteria_sync(request_obj)
        
        return { "status": "success", "result": response.evaluator_criteria.model_dump() }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def list_evaluator_info(patronus_client: PatronusAPIClient = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get combined information about evaluators and their criteria.
    Returns a dictionary with evaluator families as keys and their associated information and criteria as values.
    """
    try:
        # Get evaluators and criteria
        evaluators = _list_evaluators(patronus_client=patronus_client)
        criteria = _list_criteria(Request(data=ListCriteriaRequest()), patronus_client=patronus_client)
        
        # Create a dictionary to store results, grouped by evaluator_family
        result = {}
        
        # First, organize evaluators by family
        for evaluator in evaluators:
            family = evaluator.get('evaluator_family')
            if family not in result:
                # Remove unnecessary fields from evaluator
                evaluator_data = {k: v for k, v in evaluator.items() 
                                if k not in ['evaluator_family', 'name']}
                result[family] = {
                    'evaluator': evaluator_data,
                    'criteria': []
                }
        
        # Then, add criteria to their respective families
        for criterion in criteria:
            family = criterion.get('evaluator_family')
            if family in result:
                # Remove unnecessary fields from criterion
                criterion_data = {k: v for k, v in criterion.items() 
                                if k not in ['evaluator_family', 'revision']}
                result[family]['criteria'].append(criterion_data)
        
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def custom_evaluate(request: Request[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate using a custom evaluator function.
    Args:
        request: Request object containing:
            - evaluator_function: The evaluator function to use
            - args: List of arguments to pass to the evaluator function
    Returns:
        Dictionary containing the evaluation result
    """
    try:
        # Get the evaluator function and arguments
        evaluator_func = request.data["evaluator_function"]
        args = request.data.get("args", [])

        # Run the evaluation
        result = evaluator_func(*args)

        # Convert result to standard format
        if isinstance(result, EvaluationResult):
            return {
                "status": "success",
                "result": {
                    "score": result.score,
                    "pass_": result.pass_,
                    "text_output": result.text_output,
                    "explanation": result.explanation,
                    "metadata": result.metadata,
                    "tags": result.tags
                }
            }
        elif isinstance(result, bool):
            return {
                "status": "success",
                "result": {
                    "score": 1.0 if result else 0.0,
                    "pass_": result,
                    "text_output": "Pass" if result else "Fail",
                    "explanation": "Boolean evaluation result",
                    "metadata": {},
                    "tags": {}
                }
            }
        elif isinstance(result, (int, float)):
            return {
                "status": "success",
                "result": {
                    "score": float(result),
                    "pass_": result >= 0.7,  # Default threshold
                    "text_output": f"Score: {result}",
                    "explanation": "Numeric evaluation result",
                    "metadata": {},
                    "tags": {}
                }
            }
        elif isinstance(result, str):
            return {
                "status": "success",
                "result": {
                    "score": 1.0,
                    "pass_": True,
                    "text_output": result,
                    "explanation": "Text evaluation result",
                    "metadata": {},
                    "tags": {}
                }
            }
        else:
            return {"status": "error", "message": f"Unsupported result type: {type(result)}"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

def app_factory(patronus_api_key: str = None, patronus_api_url: str = "https://api.patronus.ai") -> FastMCP:
    """Create the MCP application"""
    patronus.init(
        api_key=patronus_api_key,
        api_url=patronus_api_url
    )

    # Initialize the client once with all required parameters
    patronus_client = PatronusAPIClient(
        api_key=patronus_api_key,
        base_url=patronus_api_url,
        client_http=httpx.Client(),
        client_http_async=httpx.AsyncClient()
    )
    
    mcp = FastMCP("patronus")
    mcp.tool()(evaluate)
    mcp.tool()(run_experiment)
    mcp.tool()(batch_evaluate)
    mcp.tool(name="list_evaluator_info")(lambda: list_evaluator_info(patronus_client=patronus_client))
    mcp.tool(name="create_criteria")(lambda request: create_criteria(request, patronus_client=patronus_client))
    mcp.tool()(custom_evaluate)

    return mcp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-key", 
        type=str,
        required=False,
        help="The Patronus API key. Can also be set via the PATRONUS_API_KEY environment variable.",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        required=False,
        help="The API URL of the Patronus API. Can also be set via the PATRONUS_API_URL environment variable.",
    )
    args = parser.parse_args()

    patronus_api_key = args.api_key or os.getenv("PATRONUS_API_KEY")
    if not patronus_api_key:
        parser.error("Patronus API Key must be provided either via --api-key argument or via the PATRONUS_API_KEY environment variable.")

    patronus_api_url = args.api_url or os.getenv("PATRONUS_API_URL") or "https://api.patronus.ai"

    app = app_factory(patronus_api_key, patronus_api_url)
    print("Starting MCP server with stdio transport")
    app.run(transport="stdio")
