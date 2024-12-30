#!/bin/bash
set -e

# BUILD_ID=6.0.0 ./build-docker-ci-remote.sh --nineteen

# Default image prefix
IMAGE_PREFIX="corcelio"

REPOSITORY="cicd"


# Check for --nineteen flag
if [[ "$*" == *"--nineteen"* ]]; then
    IMAGE_PREFIX="nineteenai"

fi

disk_usage=$(df -h --output=pcent / | tail -n 1 | tr -d ' %')

if [ "$disk_usage" -gt 50 ]; then
    echo "Disk usage is above 50%, cleaning older tags for docker images..."
    docker image ls --format "{{.Repository}}:{{.Tag}} {{.CreatedAt}}" | grep "${IMAGE_PREFIX}/cicd" | sort -rk2 > /tmp/docker_images.txt
    latest_orchestrator=$(grep 'orchestrator-' /tmp/docker_images.txt | head -n 1 | awk '{print $1}')
    latest_image_server=$(grep 'image-server-' /tmp/docker_images.txt | head -n 1 | awk '{print $1}')
    echo $latest_orchestrator >> /tmp/keep_images.txt
    echo $latest_image_server >> /tmp/keep_images.txt
    echo 'nvidia/cuda:11.8.0-devel-ubuntu20.04' >> /tmp/keep_images.txt
    cat /tmp/docker_images.txt | grep -v -F -f /tmp/keep_images.txt | awk '{print $1}' | xargs -r docker rmi -f
    docker system prune -f
else
    echo "Disk usage ($disk_usage %) is below or equal to 50%, no action needed."
fi

# Build & push images
docker build -f Dockerfile.orchestrator -t ${IMAGE_PREFIX}/cicd:orchestrator-$BUILD_ID .
docker push ${IMAGE_PREFIX}/cicd:orchestrator-$BUILD_ID

docker build -f Dockerfile.image_server -t ${IMAGE_PREFIX}/cicd:image_server-$BUILD_ID .
docker push ${IMAGE_PREFIX}/cicd:image_server-$BUILD_ID 
