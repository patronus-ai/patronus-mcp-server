import asyncio
import os
import json
import argparse
from typing import Optional, Dict, Callable
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from src.patronus_mcp.server import (
    Request, EvaluationRequest, RemoteEvaluatorConfig,
    BatchEvaluationRequest, AsyncBatchEvaluationRequest, AsyncRemoteEvaluatorConfig
)

class MCPTestClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.stdio = None
        self.write = None

    async def connect_to_server(self, server_script_path: str, api_key: Optional[str] = None):
        """Connect to an MCP server"""
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        # Get API key from provided argument, environment, or prompt
        if not api_key:
            api_key = os.getenv('PATRONUS_API_KEY')
            if not api_key:
                api_key = input("Please enter your Patronus API key: ").strip()
                if not api_key:
                    raise ValueError("Patronus API key is required")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path, "--api-key", api_key],
            env={"PATRONUS_API_KEY": api_key}
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        print("\nConnected to server with tools:", [tool.name for tool in response.tools])

    async def _handle_response(self, result, test_name: str):
        """Helper method to handle JSON response"""
        response_data = json.loads(result.content[0].text)
        print(f"\n{test_name} result:", json.dumps(response_data, indent=2))

    async def test_evaluate(self):
        """Test single evaluation"""
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
        result = await self.session.call_tool("evaluate", {"request": request.model_dump()})
        await self._handle_response(result, "Evaluation")

    async def test_batch_evaluate(self):
        """Test batch evaluation"""
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
        result = await self.session.call_tool("batch_evaluate", {"request": request.model_dump()})
        await self._handle_response(result, "Batch evaluation")

    async def test_async_batch_evaluate(self):
        """Test async batch evaluation"""
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
        result = await self.session.call_tool("async_batch_evaluate", {"request": request.model_dump()})
        await self._handle_response(result, "Async batch evaluation")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    parser = argparse.ArgumentParser(description='Run MCP tests')
    parser.add_argument('server_script', help='Path to the server script')
    parser.add_argument('--api-key', help='Patronus API key', default=None)
    args = parser.parse_args()

    client = MCPTestClient()
    try:
        await client.connect_to_server(args.server_script, args.api_key)
        
        tests: Dict[str, Callable] = {
            "1": client.test_evaluate,
            "2": client.test_batch_evaluate,
            "3": client.test_async_batch_evaluate
        }
        
        print("\nChoose a test to run:")
        for key, func in tests.items():
            print(f"{key}. {func.__name__}")
        
        choice = input("\nEnter your choice (1-3): ")
        if choice in tests:
            await tests[choice]()
        else:
            print("Invalid choice")
            
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 