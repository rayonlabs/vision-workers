from datetime import datetime
from typing import Callable, Dict
from typing import Any
from app.core import constants as cst
from app.core import models
from loguru import logger
import importlib


def _import_function(function_name: str) -> Callable | None:
    """Import a function from app.checking.functions module"""
    try:
        module_name = "app.checking.functions"
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
    except (ImportError, AttributeError) as e:
        logger.exception(f"Error importing function {function_name}")
        return None


async def score_results(
    result: models.QueryResult,
    payload: dict[str, Any],
    task_config: models.OrchestratorServerConfig,
) -> models.TaskResult:

    node_scores: Dict[int, float] = {}
    func = _import_function(task_config.checking_function)
    if func is None:
        logger.exception(f"Could not import function {task_config.checking_function}")
        return models.TaskResult(node_scores=node_scores, timestamp=datetime.now())

    if result.formatted_response is None:
        logger.info(f"Got no formatted response. Checking if we get the same fail with the same status code")
        base_score: float = await func(result, payload, task_config) # TODO : handle errors properly
        node_scores[result.node_id] = base_score
        logger.info(f"Got Axon scores: {node_scores}") 
        return models.TaskResult(node_scores=node_scores, timestamp=datetime.now())

    logger.info("Checking scores with server...")
    base_score: float = await func(result, payload, task_config) # TODO : handle errors properly

    if base_score is None:
        logger.info(f"Got no base score. Axon scores: {node_scores}")
        return models.TaskResult(node_scores=node_scores, timestamp=datetime.now())

    node_scores[result.node_id] = base_score

    logger.info(f"Got Axon scores: {node_scores}") 

    return models.TaskResult(node_scores=node_scores, timestamp=datetime.now())
