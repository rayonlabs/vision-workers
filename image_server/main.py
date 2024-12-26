from fastapi import FastAPI, HTTPException
import base_model
import inference
import traceback
from typing import Callable
from functools import wraps
from model_manager import model_manager
from starlette.responses import PlainTextResponse
from loguru import logger

app = FastAPI()


def handle_request_errors(func: Callable):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"Error in {func.__name__}: {str(e)}\n{tb_str}")
            raise HTTPException(status_code=500, detail={"error": str(e), "traceback": tb_str})
    return wrapper


@app.get("/")
async def home():
    return PlainTextResponse("Image!")


@app.post("/load_model")
@handle_request_errors
async def load_model(request_data: base_model.LoadModelRequest) -> base_model.LoadModelResponse:
    return await model_manager.download_model(request_data)


@app.post("/text-to-image")
@handle_request_errors
async def text_to_image(request_data: base_model.TextToImageBase) -> base_model.ImageResponseBody:
    return await inference.text_to_image_infer(request_data)

# @handle_request_errors
@app.post("/image-to-image")
async def image_to_image(request_data: base_model.ImageToImageBase) -> base_model.ImageResponseBody:
    return await inference.image_to_image_infer(request_data)


@app.post("/upscale")
@handle_request_errors
async def upscale(request_data: base_model.UpscaleBase) -> base_model.ImageResponseBody:
    return await inference.upscale_infer(request_data)


@app.post("/avatar")
@handle_request_errors
async def avatar(
    request_data: base_model.AvatarBase,
) -> base_model.ImageResponseBody:
    return await inference.avatar_infer(request_data)


@app.post("/inpaint")
@handle_request_errors
async def inpaint(
    request_data: base_model.InpaintingBase,
) -> base_model.ImageResponseBody:
    return await inference.inpainting_infer(request_data)


@app.post("/outpaint")
@handle_request_errors
async def outpaint(
    request_data: base_model.OutpaintingBase,
) -> base_model.ImageResponseBody:
    return await inference.outpainting_infer(request_data)


@app.post("/clip-embeddings")
@handle_request_errors
async def clip_embeddings(
    request_data: base_model.ClipEmbeddingsBase,
) -> base_model.ClipEmbeddingsResponse:
    embeddings = await inference.get_clip_embeddings(request_data)
    return base_model.ClipEmbeddingsResponse(clip_embeddings=embeddings)


@app.post("/clip-embeddings-text")
@handle_request_errors
async def clip_embeddings_text(
    request_data: base_model.ClipEmbeddingsTextBase,
) -> base_model.ClipEmbeddingsTextResponse:
    embedding = await inference.get_clip_embeddings_text(request_data)
    return base_model.ClipEmbeddingsTextResponse(text_embedding=embedding)


if __name__ == "__main__":
    import uvicorn
    import os

    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import torch

    # Below doens't do much except cause major issues with mode loading and unloading
    torch.use_deterministic_algorithms(False)

    uvicorn.run(app, host="0.0.0.0", port=6919)
