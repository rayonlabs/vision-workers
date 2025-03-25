from fastapi import FastAPI
import base_model
import inference
import traceback
from typing import Callable
from functools import wraps
from starlette.responses import PlainTextResponse
from loguru import logger
import json 
import httpx 
from typing import Dict, Any, Tuple, Union

app = FastAPI()
        
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
    return PlainTextResponse("CSM TTS!")

@app.post("/csm-tts-clone")
@handle_request_errors
async def csm_tts_clone(request_data: base_model.CSMIncoming) -> base_model.CSMResponseBody:
    try:
        synthesised_speech = await inference.csm_infer(request_data)
        logger.info("Speech response from /csm-tts-clone")
    except Exception as e:
        logger.error(f"!!!!!!!!!!!!! Error in /csm-tts-clone: {e}")
        raise e
    return synthesised_speech



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