from app.core import models
from typing import Dict, Any, Union, Tuple
import httpx
from app.core import utility_models
from app.checking import utils as checking_utils
import numpy as np
import xgboost as xgb
from loguru import logger

from app.core.constants import AI_SERVER_PORT

images_are_same_classifier = xgb.XGBClassifier()
images_are_same_classifier.load_model("image_similarity_xgb_model.json")

def validate_nsfw_consistency(miner_scores, validator_scores, rel_tolerance=1e-4, abs_tolerance=1e-3):
    return np.allclose(
        miner_scores,
        validator_scores,
        rtol=rel_tolerance,
        atol=abs_tolerance
    )

async def _get_image_similarity(
    image_response_body: utility_models.ImageResponseBody,
    expected_image_response: utility_models.ImageResponseBody,
    images_are_same_classifier: xgb.XGBClassifier,
):
    if not expected_image_response.is_nsfw:
        try:
            clip_embedding_imgb64 = await _query_endpoint_clip_embeddings({"image_b64s": [image_response_body.image_b64]})
            clip_embedding_imgb64 = clip_embedding_imgb64.clip_embeddings[0]
            clip_embedding_similiarity_internal = checking_utils.get_clip_embedding_similarity(clip_embedding_imgb64, image_response_body.clip_embeddings)

            # if the miner response has mismatches base64 image and CLIP embeddings, assign score of 0
            if clip_embedding_similiarity_internal < 0.98:
                return 0
        except:
            logger.error("Failed to query CLIP embeddings with miner's imageb64")
            return 0

    clip_embedding_similiarity = checking_utils.get_clip_embedding_similarity(image_response_body.clip_embeddings, expected_image_response.clip_embeddings)
    hash_distances = checking_utils.get_hash_distances(image_response_body.image_hashes, expected_image_response.image_hashes)

    probability_same_image_xg = images_are_same_classifier.predict_proba([hash_distances])[0][1]

    #If the miner's image is blatantly different from the expected image, assign score of -10 
    if clip_embedding_similiarity < 0.6:
        return -10

    # MODEL has a very low threshold
    score = float(probability_same_image_xg**0.5) * 0.4 + (clip_embedding_similiarity**2) * 0.6
    if score > 0.95:
        return 1

    return score**2

async def query_endpoint_with_status(endpoint: str, data: Dict[str, Any], server_name: str) -> Tuple[utility_models.ImageResponseBody | None, int]:
    url = f"http://{server_name}:{AI_SERVER_PORT}" + "/" + endpoint.lstrip("/")
    async with httpx.AsyncClient(timeout=60 * 2) as client:
        logger.info(f"Querying : {url}")
        try:
            response = await client.post(url, json=data)
            logger.info(response.status_code)
            if response.status_code >= 400:
                return None, response.status_code
            return utility_models.ImageResponseBody(**response.json()), response.status_code
        except httpx.HTTPError as e:
            status = getattr(e, 'response', None) and e.response.status_code or 500
            return None, status

async def _query_endpoint_clip_embeddings(data: Dict[str, Any]) -> utility_models.ClipEmbeddingsResponse:
    url = f"http://{models.ServerType.IMAGE.value}:{AI_SERVER_PORT}" + "/clip-embeddings"
    async with httpx.AsyncClient(timeout=60 * 2) as client:
        logger.info(f"Querying : {url}")
        response = await client.post(url, json=data)
        logger.info(response.status_code)
        return utility_models.ClipEmbeddingsResponse(**response.json())

async def check_image_result(result: models.QueryResult, payload: dict, task_config: models.OrchestratorServerConfig) -> Union[float, None]:

    # checking a fail
    if result.formatted_response is None:
        miner_status_code = result.status_code
        _, vali_status_code = await query_endpoint_with_status(task_config.endpoint, payload, task_config.server_needed.value)
        logger.info(f"miner status code: {miner_status_code} - vali status code : {vali_status_code}")
        if str(vali_status_code)[0] == str(miner_status_code)[0]:
            return 1
        else:
            return -1

    image_response_body = utility_models.ImageResponseBody(**result.formatted_response)

    if image_response_body.image_b64 is None and image_response_body.clip_embeddings is None and image_response_body.image_hashes is None and image_response_body.is_nsfw is None:
        logger.error(f"For some reason Everything is none! {image_response_body}")
        return 0

    expected_image_response, vali_status_code = await query_endpoint_with_status(task_config.endpoint, payload, task_config.server_needed.value)

    is_nsfw_payload = {
        "image": image_response_body.image_b64
    }
    try:
        vali_nsfw_scores, is_nsfw = await query_endpoint_with_status('/check-nsfw', is_nsfw_payload, task_config.server_needed.value)
        is_nsfw_score_consistent = validate_nsfw_consistency(
            vali_nsfw_scores.nsfw_scores,
            image_response_body.nsfw_scores,
        )
        if is_nsfw_score_consistent:
            is_miner_image_nsfw = is_nsfw
        else:
            logger.error(f"NSFW scores are inconsistent: {vali_nsfw_scores.nsfw_scores} vs {image_response_body.nsfw_scores}")
            is_miner_image_nsfw = False
    except Exception as e:
        logger.error(f"Failed to query NSFW endpoint: {e}")

    if expected_image_response.clip_embeddings is None:
        logger.error(f"For some reason Everything is none! {expected_image_response}")
        return None

    if is_miner_image_nsfw and is_miner_image_nsfw != image_response_body.is_nsfw:
        return -2

    else:
        return await _get_image_similarity(
            image_response_body,
            expected_image_response,
            images_are_same_classifier,
        )
