import httpx
from loguru import logger
import asyncio


async def handle_none_task_id(response: httpx.Response):
    if response.json().get("status") == "Busy":
        logger.warning("There's already a task being checked, will sleep and try again...")
        await asyncio.sleep(20)

    else:
        logger.error("Checking server seems broke, please check!")


async def handle_task_id(response: httpx.Response, task_id: str, url: str):

    async with httpx.AsyncClient() as client:
        check = 0
        while True:
            check += 1
            logger.info(f"Waiting for task {task_id} to be done - check number: {check}")
            await asyncio.sleep(1)
            task_response = await client.get(url.rstrip("/") + f"/check-task/{task_id}")
            task_response.raise_for_status()
            task_response_json = task_response.json()

            if task_response_json.get("status") != "Processing":
                if task_response_json.get("status") == "Failed":
                    logger.error(f"Task {task_id} failed: {task_response_json.get('error')}")
                    return None
                break

        logger.info(f"Task {task_id} is done: {task_response_json}")
        return task_response_json.get("result", {}), 0