import os
import subprocess
from time import sleep
import httpx
from typing import Dict, Any
import asyncio
from loguru import logger
from app.config import checking_server_configs, get_checking_server_config
from app.core.models import ServerType
from app.core.constants import AI_SERVER_PORT


class ServerManager:
    """
    This class manages starting, stopping, and handling of language and image servers.
    """

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible_devices is not None:
        gpus = cuda_visible_devices
    else:
        gpus = "all"

    docker_run_flags = f"--gpus {gpus} --runtime=nvidia"

    def __init__(self):
        self.server_process = None
        self.running_servers = {checking_server_config.name: False for checking_server_config in checking_server_configs}

    def generate_gpu_string(self, num_gpus: int) -> str:
        gpu_devices = ",".join(str(i) for i in range(num_gpus))
        return f'--gpus "device={gpu_devices}" --runtime=nvidia'

    def _kill_process_on_port(self, port):
        """
        Stop and remove Docker container running on the given port.
        """
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"publish={port}", "--format", "{{.ID}}"],
                capture_output=True,
                text=True,
                check=True,
            )
            container_id = result.stdout.strip()

            if container_id:
                subprocess.run(["docker", "stop", container_id], check=True)
                subprocess.run(["docker", "rm", container_id], check=True)
                logger.info(f"Successfully stopped and removed the container running on port {port}.")
            else:
                logger.info(f"No container is running on port {port}.")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stop the container running on port {port}: {e}")

    async def is_server_healthy(
        self,
        port: int,
        sleep_time: int = 5,
        total_attempts: int = 12 * 10 * 2,  # 20 minutes worth
        server_name: str = "localhost",
    ) -> tuple[bool, str | None]:
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
                        # Some servers don't have a health endpoint, so we try to ping the root first
                        # then fall back to this (llm has health, image does not)
                        response = await client.get(f"http://{server_name}:{port}/health")
                        server_is_healthy = response.status_code == 200

                        await asyncio.sleep(sleep_time)
                    else:
                        logger.info(f"Server {port} is healthy!")
                        return server_is_healthy, response.content.decode()
                except httpx.RequestError:
                    await asyncio.sleep(sleep_time)
                except KeyboardInterrupt:
                    break
                i += 1
                if i > total_attempts:
                    break
        return server_is_healthy, None

    async def load_model(self, load_model_config: Dict[str, Any], server_name) -> None:
        """
        Load a new model configuration
        """
        try:
            logger.debug(f"Loading model with config: {load_model_config}")
            async with httpx.AsyncClient(timeout=1200) as client:
                response = await client.post(
                    url=f"http://{server_name}:{AI_SERVER_PORT}/load_model",
                    json=load_model_config,
                )
            return response
        except httpx.HTTPError:
            raise Exception("Timeout when loading model :(")

    async def start_server(self, server_type: ServerType, load_model_config: dict | None):
        """
        Start a server with the given name.
        """
        server_config = get_checking_server_config(server_type)

        if server_config is None:
            logger.error(f"Server {server_type.value} not found in the server config")
            return

        logger.info(f"Starting server: {server_config.name}. First checking if anything is running on {server_config.port}...")
        desired_server_is_online, response_content = await self.is_server_healthy(
            port=server_config.port,
            sleep_time=1,
            total_attempts=3,
            server_name=server_config.name,
        )
        # is_server_healthy uses the server name. So if we get a 200, it was from the server we want to start, else the server
        # we want is not running
        self.running_servers[server_config.name] = desired_server_is_online

        logger.info(f"Desired server: {server_config.name}. Is running: {desired_server_is_online}.")

        if desired_server_is_online:
            if server_config.name == ServerType.LLM.value:
                correct_model_is_running = await self._check_correct_model_is_running(
                    server_name=server_config.name,
                    port=server_config.port,
                    model_name=load_model_config["model"],
                )
                if correct_model_is_running:
                    return
            # no need for image tasks
            else:
                return

        # Check no other server is running on the same port
        logger.info(f"Running servers: {self.running_servers}. Killing anything on port {server_config.port}...")
        self._kill_process_on_port(server_config.port)
        subprocess.Popen(f"docker rm -f {server_config.name}", shell=True).wait()
        for server in self.running_servers:
            self.running_servers[server] = False

        docker_run_flags = self.docker_run_flags
        if load_model_config is not None and "num_gpus" in load_model_config.keys():
            num_gpus = load_model_config["num_gpus"]
            docker_run_flags = self.generate_gpu_string(num_gpus)

        extra_docker_flags = ""
        if "extra-docker-flags" in load_model_config.keys():
            extra_docker_flags = load_model_config["extra-docker-flags"]

        sleep(2)

        shared_vol_size = os.getenv("SHARED_VOLUME_SIZE", "10g")

        command = (
            f"docker run -d --rm --shm-size={shared_vol_size} --name {server_config.name} "
            + " ".join([f"-v {volume}:{mount_path}" for volume, mount_path in server_config.volumes.items()])
            + " "
            + " ".join([f"-e {key}={val}" for key, val in server_config.env_vars.items()])
            + f" {docker_run_flags} "
            + f"-p {server_config.external_port}:{server_config.port} "
            + f"--network {server_config.network} "
            + f"{server_config.docker_image} "
        )
        if extra_docker_flags:
            command += f" {extra_docker_flags}"

        logger.info(f"Starting server: {server_config.name} 🦄")
        logger.debug(f"docker run cmd : {command}")

        self.server_process = subprocess.Popen(command, shell=True)

        server_is_up = await self.is_server_healthy(
            server_config.port,
            server_name=server_config.name,
        )
        if not server_is_up:
            raise Exception(f"Timeout when starting server {server_config.name}")

        self.running_servers[server_config.name] = True

    async def stop_server(self):
        """
        Stop the currently running server.
        """
        for server_name, is_running in self.running_servers.items():
            if is_running:
                logger.info(f"Stopping the running server container {server_name} 😈")

                try:
                    subprocess.run(["docker", "stop", server_name], check=True)
                    subprocess.run(["docker", "rm", server_name], check=True)
                    logger.info(f"Successfully stopped and removed the container: {server_name}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to stop the container {server_name}: {e}")

                self.running_servers[server_name] = False

        self.server_process = None

    def _is_container_running(self, container_name: str) -> bool:
        try:
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    f"name={container_name}",
                    "--format",
                    "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return container_name in result.stdout.split()
        except subprocess.CalledProcessError:
            return False

    @staticmethod
    async def _check_correct_model_is_running(server_name: str, port: int, model_name: str):
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"http://{server_name}:{port}/v1/models")
            if response.status_code != 200:
                logger.error(f"Server {server_name} is running, but /v1/models returned {response.status_code}???")
                logger.exception(response.text)
                return False
            model_loaded = response.json()["data"][0]["id"]
            correct_model_is_running = model_loaded == model_name
            logger.info(f"Server {server_name} is running. Model is correct: {correct_model_is_running}")
            return correct_model_is_running
