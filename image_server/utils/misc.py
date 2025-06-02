from PIL import Image
from utils import base64_utils
import base_model
import imagehash
import inference
from utils import safety_checker as sc

safety_checker = sc.Safety_Checker()


def image_hash_feature_extraction(image: Image.Image) -> base_model.ImageHashes:
    phash = str(imagehash.phash(image))
    ahash = str(imagehash.average_hash(image))
    dhash = str(imagehash.dhash(image))
    chash = str(imagehash.colorhash(image))

    return base_model.ImageHashes(
        perceptual_hash=phash,
        average_hash=ahash,
        difference_hash=dhash,
        color_hash=chash,
    )


async def take_image_and_return_formatted_response_body(
    image: Image.Image,
) -> base_model.ImageResponseBody:
    is_nsfw = safety_checker.nsfw_check(image)
    image_b64 = base64_utils.pil_to_base64(image)
    image_hashes = image_hash_feature_extraction(image)
    clip_embeddings_of_images = await inference.get_clip_embeddings(base_model.ClipEmbeddingsBase(image_b64s=[image_b64]))
    # Since we only need the first element of the list
    clip_embeddings_of_image = clip_embeddings_of_images[0]

    if is_nsfw:
        return base_model.ImageResponseBody(
            image_b64=image_b64,
            image_hashes=image_hashes,
            clip_embeddings=clip_embeddings_of_image,
            is_nsfw=is_nsfw,
        )
    else:
        return base_model.ImageResponseBody(
            image_b64=image_b64,
            image_hashes=image_hashes,
            clip_embeddings=clip_embeddings_of_image,
            is_nsfw=is_nsfw,
        )
