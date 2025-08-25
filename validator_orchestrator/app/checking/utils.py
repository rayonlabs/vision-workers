import numpy as np
from typing import List
import imagehash
from PIL import Image
import httpx
import re
from loguru import logger
from app.core import utility_models
import io
import base64



def _hash_distance(hash_1: str, hash_2: str, color_hash: bool = False) -> int:
    if color_hash:
        restored_hash1 = imagehash.hex_to_flathash(hash_1, hashsize=3)
        restored_hash2 = imagehash.hex_to_flathash(hash_2, hashsize=3)
    else:
        restored_hash1 = imagehash.hex_to_hash(hash_1)
        restored_hash2 = imagehash.hex_to_hash(hash_2)

    return restored_hash1 - restored_hash2


def get_clip_embedding_similarity(clip_embedding1: List[float], clip_embedding2: List[float]):
    image_embedding1 = np.array(clip_embedding1, dtype=float).flatten()
    image_embedding2 = np.array(clip_embedding2, dtype=float).flatten()

    norm1 = np.linalg.norm(image_embedding1)
    norm2 = np.linalg.norm(image_embedding2)

    if norm1 == 0 or norm2 == 0:
        return float(norm1 == norm2)

    dot_product = np.dot(image_embedding1, image_embedding2)
    normalized_dot_product = dot_product / (norm1 * norm2)

    return float(normalized_dot_product)


def image_hash_feature_extraction(image: Image.Image) -> utility_models.ImageHashes:
    phash = str(imagehash.phash(image))
    ahash = str(imagehash.average_hash(image))
    dhash = str(imagehash.dhash(image))
    chash = str(imagehash.colorhash(image))

    return utility_models.ImageHashes(
        perceptual_hash=phash,
        average_hash=ahash,
        difference_hash=dhash,
        color_hash=chash,
    )


def pil_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def get_hash_distances(hashes_1: utility_models.ImageHashes, hashes_2: utility_models.ImageHashes) -> List[int]:
    ahash_distance = _hash_distance(hashes_1.average_hash, hashes_2.average_hash)
    phash_distance = _hash_distance(hashes_1.perceptual_hash, hashes_2.perceptual_hash)
    dhash_distance = _hash_distance(hashes_1.difference_hash, hashes_2.difference_hash)
    chash_distance = _hash_distance(hashes_1.color_hash, hashes_2.color_hash, color_hash=True)

    return [phash_distance, ahash_distance, dhash_distance, chash_distance]


def validate_gojourney_url(url: str) -> bool:
    if not url:
        return False
    pattern = re.compile(r"^https:\/\/img\.midjourneyapi\.xyz\/mj\/.*\.png$")
    if pattern.match(url):
        return True
    return False


async def fetch_image_as_bytes(url):
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.get(url)
            return response.content
    except Exception as e:
        logger.debug(f"Error when fetching image {url}: {e}")
        return False


def validate_nsfw_consistency(miner_scores, validator_scores, rtol=5e-2, atol=2e-2):
    """Validate consistency between miner and validator scores for NSFW content."""
    for section in ("special_scores", "concept_scores"):
        m_dict = miner_scores.get(section, {})
        v_dict = validator_scores.get(section, {})
        
        shared_keys = set(m_dict.keys()) & set(v_dict.keys())
        if not shared_keys:
            continue
            
        m_values = [float(m_dict[k]) for k in shared_keys]
        v_values = [float(v_dict[k]) for k in shared_keys]

        if not np.allclose(m_values, v_values, rtol=rtol, atol=atol, equal_nan=True):
            return False
    
    return True
