import argparse
import os 
import httpx

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
import patronus, patronus.evals, patronus.experiments.experiment 
from typing import Optional, List, Dict, Any, Literal, Generic, TypeVar, Union
from patronus.api.api_client import PatronusAPIClient
from patronus.api.api_types import ListEvaluatorsResponse, ListCriteriaResponse

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

def list_evaluators(patronus_client: PatronusAPIClient = None) -> ListEvaluatorsResponse:
    """List all available evaluators from the Patronus API"""
    try:
        if not patronus_client:
            raise ValueError("Patronus client must be provided")
        response = patronus_client.list_evaluators_sync()
        evaluators = [evaluator.model_dump() for evaluator in response]
        return {"status": "success", "result": evaluators}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def list_criteria(request: Request[ListCriteriaRequest], patronus_client: PatronusAPIClient = None) -> Dict[str, Any]:
    """List all available criteria from the Patronus API"""
    try:
        if not patronus_client:
            raise ValueError("Patronus client must be provided")
        
        criteria_request = request.data if request.data else ListCriteriaRequest()
        response = patronus_client.list_criteria_sync(request=criteria_request)
        criteria = [criterion.model_dump() for criterion in response.evaluator_criteria]
        return {"status": "success", "result": criteria}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def app_factory(patronus_api_key: str, patronus_api_url: str) -> FastMCP:
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
    mcp.tool(name="list_evaluators")(lambda: list_evaluators(patronus_client=patronus_client))
    mcp.tool(name="list_criteria")(lambda request={}: list_criteria(Request(data=ListCriteriaRequest(**request)), patronus_client=patronus_client))

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
