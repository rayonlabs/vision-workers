
# Steps setup env, build and run docker image

## THIS IS TO BUID TO RUN LOCALLY, YOU WONT NEED TO DO THIS TO RUN IT. JUST FOLLOW THE READ ME IN IMAGE_SERVER/LLM_SERVER/ORCHESTRATOR

## Pre-requisties
`WITH_AUTOUPDATES=0 sudo -E ./bootstrap.sh`

### Building docker container

To build:
```bash
nohup docker build -t corcelio/vision:orchestrator-latest -f Dockerfile.orchestrator . > build.log 2>&1 &
```

To run:
```bash
docker run --gpus all --runtime=nvidia -p 6920:6920 corcelio/vision:orchestrator-latest
```

If that doesn't work:
```bash
docker run --gpus all  -p 6920:6920 corcelio/vision:orchestrator-latest
```


**Combined command**
```bash
docker build -t corcelio/vision:orchestrator-latest .
docker run --gpus all  -p 6920:6920 corcelio/vision:orchestrator-latest
```

**Combined command for local dev/ testing**
NOTE: this will use prod image server & llm server images unless otherwise specified
```bash
docker build -t corcelio/dev:orch-test . -f Dockerfile.orchestrator
docker run -p 6920:6920 -e PORT=6920 -e CUDA_VISIBLE_DEVICES=0 -e DEVICE=0 --gpus '"device=0"' --runtime=nvidia corcelio/dev:orch-test
```
or
NOTE: this will use prod image server & llm server images unless otherwise specified
```bash
docker kill orchestrator || true; docker build -t corcelio/dev:orch-test . -f Dockerfile.orchestrator; ./launch_orchestrator.sh --orchestrator-image corcelio/dev:orch-test --dont-refresh-local-images
```

**Build all**
```bash
docker build -t corcelio/dev:orch-test . -f Dockerfile.orchestrator
docker build -t corcelio/dev:image-test . -f Dockerfile.image_server
```

**Run with orch**


```bash
./launch_orchestrator.sh --orchestrator-image corcelio/dev:orch-test  --image-server-image corcelio/dev:image-test --dont-refresh-local-images
```

** Full command **
```bash
docker build -t corcelio/dev:orch-test . -f Dockerfile.orchestrator  && docker build -t corcelio/dev:image-test . -f Dockerfile.image_server && ./launch_orchestrator.sh --orchestrator-image corcelio/dev:orch-test --image-server-image corcelio/dev:image-test --dont-refresh-local-images && docker logs --tail 50 -f orchestrator
```


**Run Image Server**
```bash
docker run --name image-test --rm -v HF:/app/cache -v COMFY:/app/image_server/ComfyUI -p 6918:6919 --runtime=nvidia --gpus=all -e PORT=6919 -e DEVICE=0  corcelio/dev:image-test
```
or 
```bash
docker kill image-test || true; docker build -t corcelio/dev:image-test . -f Dockerfile.image_server; docker run  -d --rm --name image-test -v COMFY:/app/image_server/ComfyUI -v HF:/app/cache -p 6918:6918 --runtime=nvidia --gpus '"device=3"' -e PORT=6918 -e DEVICE=0 corcelio/dev:image-test; docker logs -f --tail 50 image-test
```

**Run LLM image**

```bash
docker kill llm-test || true; docker build -t corcelio/dev:llm-test . -f Dockerfile.llm_server; docker run --name llm-test -d --rm  -v HF:/app/cache -p 6918:6919 --gpus '"device=1"' --runtime=nvidia -e PORT=6919 -e MODEL=leafspark/Reflection-Llama-3.1-70B-GGUF   -e CUDA_VISIBLE_DEVICES=0 corcelio/dev:llm-test; docker logs -f --tail 50 llm-test
```

### Uploading to docker hub
Get your credentials ready for docker hub
```bash
docker login
```

```bash
docker push corcelio/vision:orchestrator-latest
```


### Trouble shooting
If you run into any Cuda prbs during build/run (especially with the argyment --runtime=nvidia), make sure to do this, then restart the last steps :
```bash
sudo nano /etc/docker/daemon.json
```

-> Change it to this : (define the default-runtime)
```json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

### Install stuff to run orchestrator without docker

[Intstall miniconda](./install_miniconda_create_venv.md)

```bash
sudo apt-get update && \
sudo DEBIAN_FRONTEND=noninteractive TZ=Europe/London apt-get install -y \
    wget \
    git \
    curl \
    lsof \
    python3-dev \
    build-essential \
    python3-pip \
    apt-utils \
    vim \
    sudo \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3-tk \
    python3-dev \
    git-lfs \
    unzip && \
sudo ln -snf /usr/share/zoneinfo/Europe/London /etc/localtime && \
echo "Europe/London" | sudo tee /etc/timezone && \
sudo apt-get clean && \
sudo rm -rf /var/lib/apt/lists/*
```

```bash
LLM_SERVER_DOCKER_IMAGE=corcelio/dev:llm-test IMAGE_SERVER_DOCKER_IMAGE=corcelio/dev:image-test  ./entrypoint.sh 
```