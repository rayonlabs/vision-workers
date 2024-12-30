#!/bin/bash
ORCHESTRATOR_IMAGE="nineteenai/sn19:orchestrator-latest"
IMAGE_SERVER_IMAGE="nineteenai/sn19:image_server-latest"
PORT=6920
REFRESH_LOCAL_IMAGES=1

# ./launch_orchestrator.sh --image-server-image nineteenai/sn19:image_server-6.0.0 --orchestrator-image nineteenai/sn19:orchestrator-6.0.0

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
  --nvidia-runtime) NVIDIA_RUNTIME_FLAG=1 ;;
  --device)
    DEVICE="$2"
    shift
    ;;
  --orchestrator-image)
    ORCHESTRATOR_IMAGE="$2"
    shift
    ;;
  --image-server-image)
    IMAGE_SERVER_IMAGE="$2"
    shift
    ;;
  --port)
    PORT="$2"
    shift
    ;;
  --dont-refresh-local-images)
    REFRESH_LOCAL_IMAGES=0
    ;;
  *)
    echo "Unknown parameter: $1"
    exit 1
    ;;
  esac
  shift
done


NETWORK="comm"
ORCHESTRATOR_CONTAINER_NAME="orchestrator"

DOCKER_RUN_FLAGS="--rm \
                  -v /var/run/docker.sock:/var/run/docker.sock \
                  -e LLM_SERVER_DOCKER_IMAGE=$LLM_IMAGE \
                  -e IMAGE_SERVER_DOCKER_IMAGE=$IMAGE_SERVER_IMAGE \
                  --network $NETWORK"

# Add the --runtime=nvidia flag unless --no-runtime-flag is specified

if [[ -n "$NVIDIA_RUNTIME_FLAG" ]]; then
  DOCKER_RUN_FLAGS+=" --runtime=nvidia"
fi

if [[ -n "$DEVICE" ]]; then
  DOCKER_RUN_FLAGS+=" -e CUDA_VISIBLE_DEVICES=$DEVICE \
                        -e DEVICE=$DEVICE \
                        --gpus \"device=$DEVICE\""
else
  DOCKER_RUN_FLAGS+=" --gpus all"
fi

check_and_pull_image() {
  IMAGE=$1

  get_local_image_digest() {
    DIGEST=$(docker image inspect "$IMAGE" --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}' 2>/dev/null | cut -d':' -f2)
    echo $DIGEST
  }

  LOCAL_DIGEST=$(get_local_image_digest)

  if [ -z "$LOCAL_DIGEST" ]; then
    echo "Image $IMAGE does not exist locally. Attempting to pull from Docker Hub..."
    if docker pull $IMAGE; then
      echo "Successfully pulled $IMAGE"
    else
      echo "Failed to pull $IMAGE. Please check the image name and your network connection."
      return 1
    fi
  else
    echo "Image $IMAGE exists locally. Checking for updates..."
    if [ "$REFRESH_LOCAL_IMAGES" -eq 1 ]; then
      if docker pull $IMAGE; then
        echo "Successfully pulled latest version of $IMAGE"
      else
        echo "Failed to pull latest version of $IMAGE. Using existing local image."
      fi
    fi
    REMOTE_DIGEST=$(get_local_image_digest)
    if [ "$REMOTE_DIGEST" != "$LOCAL_DIGEST" ]; then
      echo "A new version of the image $IMAGE is available and has been pulled."
    else
      echo "The image $IMAGE is up-to-date."
    fi
  fi
}
check_and_pull_image $ORCHESTRATOR_IMAGE
check_and_pull_image $IMAGE_SERVER_IMAGE

echo "Got up to date images, Making volumes...."

docker volume inspect HF >/dev/null 2>&1 || docker volume create HF
docker volume inspect COMFY >/dev/null 2>&1 || docker volume create COMFY

docker network inspect $NETWORK >/dev/null 2>&1 || docker network create $NETWORK

echo "Volumes & networks created. Launching orchestrator..."

if [ -n "$(docker ps -q -f name=$ORCHESTRATOR_CONTAINER_NAME)" ]; then
  echo "Container $ORCHESTRATOR_CONTAINER_NAME found. Stopping and removing it..."
  docker stop $ORCHESTRATOR_CONTAINER_NAME 2>/dev/null || true
  echo "Container stopped. Removing in 5..."
  sleep 5
  docker rm -f "$ORCHESTRATOR_CONTAINER_NAME" 2>/dev/null || true
  echo "Container removed. Launching in 5..."
  sleep 5
fi

# TODO: remove later

docker stop llm_server 2>/dev/null || true
sleep 5
docker rm -f llm_server 2>/dev/null || true

docker run -d --rm --name $ORCHESTRATOR_CONTAINER_NAME $DOCKER_RUN_FLAGS -e PORT=$PORT -p $PORT:$PORT $ORCHESTRATOR_IMAGE