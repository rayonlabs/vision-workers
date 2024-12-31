#!/bin/bash

# Example
# ./update-docker-images.sh -o nineteenai/cicd:orchestrator-6.6.0 -i nineteenai/cicd:image_server-6.6.0  -s 6.6.0

usage() {
    echo "Usage: $0 [-o ORCHESTRATOR_IMAGE] [-l LLM_IMAGE] [-i IMAGE_SERVER_IMAGE]"
    echo "At least one of -o, -l, or -i must be specified."
    echo "Optional: -d DESTINATION_PREFIX (default: nineteenai/sn19)"
    echo "Optional: -s SUFFIX (default: latest)"
    exit 1
}

# Default values
DEST_PREFIX="nineteenai/sn19"
SUFFIX="latest"

while getopts "o:l:i:d:s:" opt; do
    case $opt in
        o) ORCHESTRATOR_IMAGE="$OPTARG" ;;
        l) LLM_IMAGE="$OPTARG" ;;
        i) IMAGE_SERVER_IMAGE="$OPTARG" ;;
        d) DEST_PREFIX="$OPTARG" ;;
        s) SUFFIX="$OPTARG" ;;
        *) usage ;;
    esac
done

if [ -z "$ORCHESTRATOR_IMAGE" ] && [ -z "$LLM_IMAGE" ] && [ -z "$IMAGE_SERVER_IMAGE" ]; then
    echo "Error: At least one of ORCHESTRATOR_IMAGE, LLM_IMAGE, or IMAGE_SERVER_IMAGE must be specified."
    usage
fi

# Prompt for Docker Hub credentials
read -p "Docker Hub Username: " DOCKER_USER
read -s -p "Docker Hub Password: " DOCKER_PASS
echo

# Login to Docker Hub
echo "Logging in to Docker Hub..."
echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin
if [ $? -ne 0 ]; then
    echo "Error: Docker login failed"
    exit 1
fi
 
tag_and_push() {
    local SOURCE_IMAGE=$1
    local DEST_TAG=$2

    echo "Pulling $SOURCE_IMAGE"
    docker pull $SOURCE_IMAGE
    if [ $? -ne 0 ]; then
        echo "Error: Failed to pull $SOURCE_IMAGE"
        return 1
    fi

    echo "Tagging $SOURCE_IMAGE as $DEST_TAG"
    docker tag $SOURCE_IMAGE $DEST_TAG
    if [ $? -ne 0 ]; then
        echo "Error: Failed to tag $SOURCE_IMAGE as $DEST_TAG"
        return 1
    fi

    echo "Pushing $DEST_TAG"
    docker push $DEST_TAG
    if [ $? -ne 0 ]; then
        echo "Error: Failed to push $DEST_TAG"
        return 1
    fi

    return 0
}
process_image() {
    local IMAGE=$1
    local TYPE=$2
    local DEST_TAG

    if [ "$TYPE" = "Orchestrator" ]; then
        DEST_TAG="$DEST_PREFIX:${TYPE,,}-$SUFFIX"
    else
        DEST_TAG="$DEST_PREFIX:${TYPE,,}_server-$SUFFIX"
    fi

    if [ -n "$IMAGE" ]; then
        tag_and_push $IMAGE $DEST_TAG
        local RESULT=$?
        if [ $RESULT -eq 0 ]; then
            echo "$TYPE image ($IMAGE) tagged and pushed as:"
            echo "  $DEST_TAG"
            echo "Source: $IMAGE -> Destination: $DEST_TAG"
        else
            echo "Failed to process $TYPE image ($IMAGE)"
        fi
        return $RESULT
    fi
    return 0
}

echo "Summary of actions:"
process_image "$ORCHESTRATOR_IMAGE" "Orchestrator"
ORCHESTRATOR_RESULT=$?

process_image "$LLM_IMAGE" "LLM"
LLM_RESULT=$?

process_image "$IMAGE_SERVER_IMAGE" "Image"
IMAGE_SERVER_RESULT=$?

echo "-------------------------------------"
echo "Summary:"
echo "-------------------------------------"
[ -n "$ORCHESTRATOR_IMAGE" ] && [ $ORCHESTRATOR_RESULT -eq 0 ] && echo "Source: $ORCHESTRATOR_IMAGE -> Destination: $DEST_PREFIX:orchestrator-$SUFFIX"
[ -n "$LLM_IMAGE" ] && [ $LLM_RESULT -eq 0 ] && echo "Source: $LLM_IMAGE -> Destination: $DEST_PREFIX:llm_server-$SUFFIX"
[ -n "$IMAGE_SERVER_IMAGE" ] && [ $IMAGE_SERVER_RESULT -eq 0 ] && echo "Source: $IMAGE_SERVER_IMAGE -> Destination: $DEST_PREFIX:image_server-$SUFFIX"
echo "-------------------------------------"

if [ $ORCHESTRATOR_RESULT -ne 0 ] || [ $LLM_RESULT -ne 0 ] || [ $IMAGE_SERVER_RESULT -ne 0 ]; then
    echo "Some images failed to process."
    exit 1
fi

echo "All specified images processed successfully."


