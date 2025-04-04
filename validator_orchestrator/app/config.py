from pydantic import BaseModel
from typing import Dict
import os
from app.core.models import ServerType, ProdDockerImages

DEFAULT_NETWORK_NAME = "comm"


class CheckingServerConfig(BaseModel):
    name: str
    docker_image: str
    port: int
    volumes: Dict[str, str]  # volue name, mount path
    env_vars: Dict[str, str]
    network: str
    external_port: int


shared_network = os.getenv("SHARED_NETWORK_NAME", DEFAULT_NETWORK_NAME)

checking_server_configs: list[CheckingServerConfig] = [
    CheckingServerConfig(
        name=ServerType.LLM.value,
        docker_image="vllm/vllm-openai:v0.6.4.post1",
        port=6919,
        volumes={
            "HF": "/app/cache",
        },
        env_vars={
            "HF_HOME": "/app/cache",
            "HF_HUB_CACHE": "/app/cache/hub"
        },
        network=shared_network,
        external_port=6919,
    ),
    CheckingServerConfig(
        name=ServerType.IMAGE.value,
        docker_image=os.getenv("IMAGE_SERVER_DOCKER_IMAGE", ProdDockerImages.IMAGE),
        port=6919,
        volumes={"COMFY": "/app/image_server/ComfyUI"},
        env_vars={},
        network=shared_network,
        external_port=6919,
    ),
]


def get_checking_server_config(server_type: ServerType) -> CheckingServerConfig | None:
    for worker_config in checking_server_configs:
        if worker_config.name == server_type.value:
            return worker_config
    return None
