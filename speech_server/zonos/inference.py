from typing import List
import base_model
import torch
import os
from loguru import logger
import json
from typing import Dict, Any, Tuple, Union
import httpx

async def query_endpoint_with_status(
    data: Dict[str, Any],
    endpoint: str = "/generate",
    base_url: str = "http://localhost:5005/"
) -> Tuple[Union[Dict[str, Any], None], int]:
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    
    async with httpx.AsyncClient(timeout=20) as client:
        logger.info(f"Querying: {url}")
        try:
            response = await client.post(url, json=data)
            logger.info(f"Status code: {response.status_code}")
            
            if response.status_code >= 400:
                return None, response.status_code
                
            try:
                response_data = response.json()
                if isinstance(response_data, str):
                    response_data = json.loads(response_data)
                return response_data, response.status_code
                
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON response: {e}")
                return None, response.status_code
                
        except httpx.HTTPError as e:
            status = getattr(e, 'response', None)
            status_code = status.status_code if status else 500
            logger.error(f"HTTP error occurred: {e}")
            return None, status_code

async def csm_infer(
    infer_props: base_model.CSMIncoming,
) -> base_model.CSMResponseBody:
    payload = None 
    try:
        infer_props_dict = json.loads(infer_props.json())
        payload = {x:y for x,y in infer_props_dict.items() if y is not None}
    except Exception as e:
        logger.error(f"Error in getting enhanced response: {e}")
        logger.info(f"infer_props for enhance request: {infer_props_dict.items()}")

    speech_response, status_code = await query_endpoint_with_status(payload)
    return speech_response