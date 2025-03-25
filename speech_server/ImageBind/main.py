from fastapi import FastAPI
import base_model
import inference
import traceback
from typing import Callable
from functools import wraps
from starlette.responses import PlainTextResponse
from utils.checking_server_gateway import get_checking_gateway_instance, SPEECH_TASK_TO_PORT_MAPPING
from loguru import logger

app = FastAPI()
container_gateway = get_checking_gateway_instance()

def handle_request_errors(func: Callable):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            tb_str = traceback.format_exc()
            return {"error": str(e), "traceback": tb_str}

    return wrapper

@app.get("/")
async def home():
    return PlainTextResponse("ImageBind!")

@app.post("/imagebind-embeddings")
@handle_request_errors
async def imagebind_embeddings(
    request_data: base_model.ImagebindEmbeddingsBase,
) -> base_model.ImagebindEmbeddingsResponse:
    try:
        embeddings = await inference.get_imagebind_embeddings(request_data)
    except Exception as e:

        raise e
    return embeddings


if __name__ == "__main__":
    import uvicorn
    import os

    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import torch

    # Below doens't do much except cause major issues with mode loading and unloading
    torch.use_deterministic_algorithms(False)

    uvicorn.run(app, host="0.0.0.0", port=6919)
    print('Speech server started on port 6919...')
