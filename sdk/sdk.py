"""
SDK for the orchestrator to make it a bit easier to use

Example usage:

Go into an ipynb, for example, and run:


```
%load_ext autoreload
%autoreload 2

from sdk import sdk
await sdk.check_result(task="chat-llama-3-2-3b", orchestrator_url="http://94.156.8.113:6921")
```

"""

from typing import Any

import httpx
from sdk.orchestrator_handling import handle_none_task_id, handle_task_id
from sdk.task_config import task_configs_factory
from sdk.logging import get_logger

logger = get_logger(__name__)

example_payload = {
    "messages": [{"role": "user", "content": "Tell me the capital of France in 20 words only."}],
    "temperature": 0.2,
    "max_tokens": 500,
    "model": "chat-llama-3-2-3b",
    "top_p": 1,
    "top_k": 5,
    "stream": True,
    "logprobs": True,
    "seed": 42,
}

example_completion_payload = {
    "prompt": "Tell me the capital of France in 20 words only.",
    "temperature": 0.5,
    "max_tokens": 500,
    "model": "chat-llama-3-2-3b",
    "top_p": 1,
    "top_k": 5,
    "logprobs": True,
    "stream": True,
    "seed": 42,
}


example_response = [
    {
        "choices": [{"index": 0, "delta": {"content": "Great"}, "logprobs": {"content": [{"logprob": -1.9118125438690186}]}}],
    },
    {
        "choices": [{"index": 0, "delta": {"content": " Great"}, "logprobs": {"content": [{"logprob": -1.9118125438690186}]}}],
    },
    {
        "choices": [{"index": 0, "delta": {"content": " Great"}, "logprobs": {"content": [{"logprob": -1.9118125438690186}]}}],
    },
    {
        "choices": [{"index": 0, "delta": {"content": " Great"}, "finish_reason": "stop", "logprobs": {"content": [{"logprob": -1.9118125438690186}]}}],
    },
]


# async def _handle_stream(response: httpx.Response) -> dict[str, Any]:
    

async def get_real_response(task: str, payload: dict[str, Any] = example_payload, llm_url: str | None = None, chat: bool = True) -> dict[str, Any]:
    if llm_url is None:
        logger.warning("No LLM URL provided, using default")
        llm_url = "http://llm_server:6919"
    async with httpx.AsyncClient() as client:
        if chat:
            response = await client.post(llm_url.rstrip("/") + "/chat/completions", json=payload)
        else:
            response = await client.post(llm_url.rstrip("/") + "/completions", json=payload)
        response.raise_for_status()
        return response.json()


async def check_result(
    task: str, payload: dict[str, Any] = example_payload, miner_response: dict[str, Any] = example_response, response_time: float = 1.0, orchestrator_url: str | None = None
) -> bool:
    task_configs = task_configs_factory()
    task_config = task_configs[task]
    if orchestrator_url is None:
        logger.warning("No orchestrator URL provided, using default")
        orchestrator_url = "http://localhost:6920"

    actual_payload = {
        "server_config": {
            "server_needed": task_config.orchestrator_server_config.server_needed,
            "load_model_config": task_config.orchestrator_server_config.load_model_config,
            "checking_function": task_config.orchestrator_server_config.checking_function,
            "task": task,
            "endpoint": task_config.orchestrator_server_config.endpoint,
        },
        "result": {"formatted_response": miner_response, "node_id": 0, "response_time": response_time},
        "payload": payload,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(orchestrator_url.rstrip("/") + "/check-result", json=actual_payload)
        if response.status_code == 422:
            logger.error(f"Request failed due to {response.status_code}: {response.json().get('detail')}")
        response.raise_for_status()
        task_id = response.json().get("task_id")
        logger.info(f"Got task ID: {task_id} !!!")
        if task_id is None:
            await handle_none_task_id(response)
        else:
            task_result = await handle_task_id(response, task_id, orchestrator_url)
            logger.info(f"Task {task_id} is done: {task_result}")
            return task_result
