import asyncio
from typing import Dict, Any
from chutes.image import Image
from chutes.chute import Chute, NodeSelector

image = Image(
    username="nineteen", 
    name="image-server", 
    tag="latest", 
    readme="## Nineteen.ai image server - AI image generation service"
).from_base("nineteenai/sn19:image_server-latest")

chute = Chute(
    username="nineteen",
    name="image-server",
    readme="""## Nineteen.ai Image Server  """,
    image=image,
    concurrency=1,  
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=40
    ),
)

@chute.on_startup()
async def initialize(self):
    import subprocess
    import os
    import time
    
    os.environ["DEVICE"] = "0"
    os.environ["VRAM_MODE"] = "--normalvram"
    os.environ["WARMUP"] = "false"
    os.environ["PORT"] = "6919" 
    
    print("Starting image server...")
    subprocess.Popen(["bash", "-c", "cd /app/image_server && ./entrypoint.sh"])
    
    time.sleep(30)  
    print("Image server should be running now")

@chute.cord(public_api_path="/load_model", public_api_method="POST")
async def load_model(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    import requests
    
    try:
        response = requests.post(
            "http://localhost:6919/load_model",
            json=request_data,
            timeout=300
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise Exception(f"Failed to load model: {str(e)}")

@chute.cord(public_api_path="/text-to-image", public_api_method="POST")
async def text_to_image(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    import requests
    
    try:
        response = requests.post(
            "http://localhost:6919/text-to-image",
            json=request_data,
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error in text-to-image: {str(e)}")
        raise Exception(f"Failed in text-to-image: {str(e)}")

@chute.cord(public_api_path="/image-to-image", public_api_method="POST")
async def image_to_image(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    import requests
    
    try:
        response = requests.post(
            "http://localhost:6919/image-to-image",
            json=request_data,
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error in image-to-image: {str(e)}")
        raise Exception(f"Failed in image-to-image: {str(e)}")

@chute.cord(public_api_path="/upscale", public_api_method="POST")
async def upscale(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    import requests
    
    try:
        response = requests.post(
            "http://localhost:6919/upscale",
            json=request_data,
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error in upscale: {str(e)}")
        raise Exception(f"Failed in upscale: {str(e)}")

@chute.cord(public_api_path="/avatar", public_api_method="POST")
async def avatar(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    import requests
    
    try:
        response = requests.post(
            "http://localhost:6919/avatar",
            json=request_data,
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error in avatar generation: {str(e)}")
        raise Exception(f"Failed in avatar generation: {str(e)}")

@chute.cord(public_api_path="/inpaint", public_api_method="POST")
async def inpaint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    import requests
    
    try:
        response = requests.post(
            "http://localhost:6919/inpaint",
            json=request_data,
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error in inpainting: {str(e)}")
        raise Exception(f"Failed in inpainting: {str(e)}")

@chute.cord(public_api_path="/outpaint", public_api_method="POST")
async def outpaint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    import requests
    
    try:
        response = requests.post(
            "http://localhost:6919/outpaint",
            json=request_data,
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error in outpainting: {str(e)}")
        raise Exception(f"Failed in outpainting: {str(e)}")

@chute.cord(public_api_path="/clip-embeddings", public_api_method="POST")
async def clip_embeddings(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    import requests
    
    try:
        response = requests.post(
            "http://localhost:6919/clip-embeddings",
            json=request_data,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error in CLIP embeddings extraction: {str(e)}")
        raise Exception(f"Failed in CLIP embeddings extraction: {str(e)}")

@chute.cord(public_api_path="/clip-embeddings-text", public_api_method="POST")
async def clip_embeddings_text(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    import requests
    
    try:
        response = requests.post(
            "http://localhost:6919/clip-embeddings-text",
            json=request_data,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error in text CLIP embeddings extraction: {str(e)}")
        raise Exception(f"Failed in text CLIP embeddings extraction: {str(e)}")

@chute.cord(public_api_path="/health", public_api_method="GET")
async def health_check(self) -> Dict[str, Any]:
    import requests
    try:
        response = requests.get("http://localhost:6919/", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "message": "Image server is running"}
        else:
            return {"status": "unhealthy", "message": f"Image server returned status code {response.status_code}"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"Error connecting to image server: {str(e)}"}