from base_model import ImagebindEmbeddingsBase, ImagebindEmbeddingsResponse
import httpx
from enum import Enum
from typing import Dict, Any
import asyncio
from loguru import logger
from pydantic import BaseModel

class Task(Enum):
    imagebind_audio_embeddings = "imagebind-audio-embeddings"

SPEECH_TASK_TO_PORT_MAPPING = {
    Task.imagebind_audio_embeddings: 5001
}

class CheckingServerGateway:
    def __init__(self) -> None:
        self.task_to_endpoint = {
            Task.imagebind_audio_embeddings: f"http://localhost:{SPEECH_TASK_TO_PORT_MAPPING[Task.imagebind_audio_embeddings]}/predictions",
        }
        self.task_to_input_model = {
            Task.imagebind_audio_embeddings: ImagebindEmbeddingsBase,
        }
        self.task_to_output_model = {
            Task.imagebind_audio_embeddings: ImagebindEmbeddingsResponse,
        }
    
    async def query_task_endpoint(
            self, task: Task, data: Dict[str, Any]
        ) -> Dict[str, Any]:
        logger.info(f"Container for task {task} is ready to query")

        endpoint = self.task_to_endpoint[task]
        logger.info(f"Querying container for task {task} from endpoint {endpoint}")
        async with httpx.AsyncClient(timeout=60) as client:
            logger.info("before start time!!!")
            start_time = asyncio.get_event_loop().time()
            logger.info("after start time!!!")
            while True:
                try:
                    response = await client.post(endpoint, json=data)
                    logger.info(f"Received POST response from container for task {task} from endpoint {endpoint}")
                    if response.status_code == 200:
                        return response.json()
                    elif response.status_code == 409 and response.json().get('detail') == "Already running a prediction":
                        # Check if the timeout has been reached
                        logger.info("Container busy with another query smhhh... 🤦‍♂️")
                        if asyncio.get_event_loop().time() - start_time >= 10:
                            raise TimeoutError("Request timed out after 10 seconds")
                        # Wait a bit before retrying
                        await asyncio.sleep(0.5)
                    else:
                        response.raise_for_status()
                except Exception as e:
                    # Handle other HTTP errors if necessary
                    logger.info(f"Error: {e}")
                    raise e
                
    async def is_server_healthy(
        self,
        port: int,
        server_name: str,
        sleep_time: int = 5,
        total_attempts: int = 12 * 10 * 2,  # 20 minutes worth
    ) -> bool:
        """
        Check if server is healthy.
        """
        server_is_healthy = False
        i = 0
        await asyncio.sleep(sleep_time)
        async with httpx.AsyncClient(timeout=5) as client:
            while not server_is_healthy:
                try:
                    logger.info("Pinging " + f"http://{server_name}:{port}")
                    response = await client.get(f"http://{server_name}:{port}")
                    server_is_healthy = response.status_code == 200
                    if not server_is_healthy:
                        await asyncio.sleep(sleep_time)
                    else:
                        return server_is_healthy
                except httpx.RequestError:
                    await asyncio.sleep(sleep_time)
                except KeyboardInterrupt:
                    break
                i += 1
                if i > total_attempts:
                    return server_is_healthy
        return server_is_healthy


    async def query_container(self, task: Task, data: dict) -> BaseModel:
        server_is_up = await self.is_server_healthy(
            port= SPEECH_TASK_TO_PORT_MAPPING[task],
            server_name="localhost",
            sleep_time=1,
            total_attempts=3,
        )

        if server_is_up:
            #Send query to the container only if it is healthy
            response = await self.query_task_endpoint(task, data)
            parsed_response = self.task_to_output_model[task](**response)
            return parsed_response
        else:
            raise Exception(f"BentoML worker for task {task} is not healthy, ded :(")


checking_server_gateway = None 

def get_checking_gateway_instance() -> CheckingServerGateway:
    global checking_server_gateway
    if checking_server_gateway is None:
        checking_server_gateway = CheckingServerGateway()
    return checking_server_gateway