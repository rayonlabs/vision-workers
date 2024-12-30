from __future__ import annotations
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime

AxonScores = Dict[int, float]


class QueryResult(BaseModel):
    formatted_response: dict[str, Any] | list[dict[str, Any]] | None
    node_id: Optional[int]
    response_time: Optional[float]


class ChatTokens(BaseModel):
    token: str


class ServerType(Enum):
    LLM = "llm_server"
    IMAGE = "image_server"


class ProdDockerImages(Enum):
    LLM = "corcelio/vision:llm_server-latest"
    IMAGE = "corcelio/vision:image_server-latest"


class ModelConfigDetails(BaseModel):
    model: str
    tokenizer: Optional[str] = None
    half_precision: Optional[bool] = None
    revision: Optional[str] = None
    gpu_memory_utilization: Optional[float] = 0.8
    max_model_len: Optional[int] = None


class OrchestratorServerConfig(BaseModel):
    server_needed: ServerType = Field(examples=[ServerType.LLM, ServerType.IMAGE])
    load_model_config: dict | None = Field(
        examples=[
            {
                "model": "unsloth/Meta-Llama-3.1-8B-Instruct",
                "half_precision": True,
                "tokenizer": "tau-vision/llama-tokenizer-fix",
                "max_model_len": 16000,
                "gpu_memory_utilization": 0.6,
                "eos_token_id": 128009
            },
            None,
        ]
    )
    checking_function: str = Field(examples=["check_text_result", "check_image_result"])
    task: str = Field(examples=["chat-llama-3-1-8b"])
    endpoint: str = Field(examples=["/generate_text"])


class CheckResultsRequest(BaseModel):
    server_config: OrchestratorServerConfig
    result: QueryResult
    payload: dict


class Message(BaseModel):
    role: str
    content: str


class MessageResponse(BaseModel):
    content: str
    logprob: float


class ChatRequestModel(BaseModel):
    messages: list[Message]
    seed: int
    temperature: float
    max_tokens: int
    top_k: int
    number_of_logprobs: int
    starting_assistant_message: bool

class CompletionRequestModel(BaseModel):
    prompt: str
    seed: int
    temperature: float
    max_tokens: int
    top_k: int
    number_of_logprobs: int
    
class MinerChatResponse(BaseModel):
    text: str
    logprob: float


# TODO: These prob need refactoring - also test models should not be here


# class ValidationTest(BaseModel):
#     validator_server: ServerInstance
#     validator_task: str
#     miners_to_test: List[ServerInstance]
#     prompts_to_check: List[ChatRequestModel]
#     checking_function: Callable[
#         [QueryResult, Dict[str, Any], TaskConfig],
#         Coroutine[Any, Any, Union[float, None]],
#     ]


class TestInstanceResults(BaseModel):
    task: str
    score: float
    miner_server: ServerInstance
    validator_server: ServerInstance
    messages: List[Message]
    temperature: float
    seed: int
    miner_request: Any


class ServerDetails(BaseModel):
    id: str
    cuda_version: str
    gpu: str
    endpoint: str


class ServerInstance(BaseModel):
    server_details: ServerDetails
    model: Optional[ModelConfigDetails]


class Logprob(BaseModel):
    index: int
    logprob: float
    decoded: str


class ValidatorCheckingResponse(BaseModel):
    choices: List[ValidatorCheckingResponseBody]


class ValidatorCheckingResponseBody(BaseModel):
    delta: DeltaContent
    logprobs: LogprobContent


class DeltaContent(BaseModel):
    content: str


class LogprobContent(BaseModel):
    content: List[Logprob]


class CheckResultResponse(BaseModel):
    task_id: Union[str, None]
    status: TaskStatus


class TaskResultResponse(BaseModel):
    task_id: str
    result: Union[Dict, str]


class CheckTaskResponse(BaseModel):
    task_id: str
    result: Optional[TaskResult] = None
    status: TaskStatus


class TaskResult(BaseModel):
    node_scores: Optional[AxonScores] = None
    timestamp: datetime
    error_message: Optional[str] = None
    traceback: Optional[str] = None


class TaskStatus(Enum):
    Processing = "Processing"
    Success = "Success"
    Failed = "Failed"
    Busy = "Busy"
    Missing = "Missing"


class AllTaskStatusResponse(BaseModel):
    tasks: Dict[str, TaskStatus]
