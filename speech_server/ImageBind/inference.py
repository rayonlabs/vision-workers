from typing import List
import base_model
import constants as cst
from payload import PayloadModifier
import torch
import os
from utils.checking_server_gateway import get_checking_gateway_instance, Task
from loguru import logger
import json


payload_modifier = PayloadModifier()
checking_server_gateway = get_checking_gateway_instance()

async def get_imagebind_embeddings(
    infer_props: base_model.ImagebindEmbeddingsBase,
) -> base_model.ImagebindEmbeddingsResponse:
    payload = None
    try:
        infer_props_dict = json.loads(infer_props.json())
        payload = {x:y for x,y in infer_props_dict.items() if y is not None}
    except Exception as e:
        logger.error(f"Error in getting imagebind embeddings: {e}")
        logger.info(f"infer_props for imagebind embeddings: {infer_props_dict.items()}")

    imagebind_embedding_response = await checking_server_gateway.query_container(Task.imagebind_audio_embeddings, payload)
    return imagebind_embedding_response