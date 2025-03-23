from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
import patronus, patronus.evals, patronus.experiments.experiment 
from typing import Optional, List, Dict, Any, Literal, Generic, TypeVar, Union
import os

mcp = FastMCP("patronus")

T = TypeVar('T')

class Request(BaseModel, Generic[T]):
    data: T

class InitRequest(BaseModel):
    api_key: Optional[str] = None
    project_name: Optional[str] = None
    app: Optional[str] = None

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
    evaluators: List[RemoteEvaluatorConfig]
    task_input: Optional[str] = None
    task_output: Optional[str] = None
    system_prompt: Optional[str] = None
    task_context: Union[list[str], str, None] = None
    task_attachments: Union[list[Any], None] = None
    gold_answer: Optional[str] = None
    task_metadata: Optional[Dict[str, Any]] = None

class AsyncBatchEvaluationRequest(BaseModel):
    evaluators: List[AsyncRemoteEvaluatorConfig]
    task_input: Optional[str] = None
    task_output: Optional[str] = None
    system_prompt: Optional[str] = None
    task_context: Union[list[str], str, None] = None
    task_attachments: Union[list[Any], None] = None
    gold_answer: Optional[str] = None
    task_metadata: Optional[Dict[str, Any]] = None

@mcp.tool()
async def initialize(request: Request[InitRequest]):
    try:
        patronus.init(
            project_name=request.data.project_name,
            api_key=request.data.api_key or os.getenv("PATRONUS_API_KEY"),
            app=request.data.app,
        )
        return {"status": "success", "message": f"Patronus initialized with project: {request.data.project_name}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

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

@mcp.tool()
async def evaluate(request: Request[EvaluationRequest]):
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

@mcp.tool()
async def run_experiment(request: Request[ExperimentRequest]):
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

@mcp.tool()
async def batch_evaluate(request: Request[BatchEvaluationRequest]):
    try:
        if not request.data.evaluators:
            return {"status": "error", "message": "No evaluators provided"}
            
        evaluators = [
            _create_evaluator(config)
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
        
        with patronus.Patronus() as client:
            results = client.evaluate(
                evaluators=evaluators,
                **eval_kwargs
            )
            print([eval for eval in results.succeeded_evaluations()])
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

@mcp.tool()
async def async_batch_evaluate(request: Request[AsyncBatchEvaluationRequest]):
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

if __name__ == "__main__":
    print("Starting MCP server with stdio transport")
    mcp.run(transport="stdio")
