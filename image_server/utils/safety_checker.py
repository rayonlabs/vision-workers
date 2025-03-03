import numpy as np
from PIL import Image
from functools import partial
import torch
from torch import nn
from diffusers import StableDiffusionPipeline
from typing import Tuple
import constants as cst
import os


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


@torch.no_grad()
def forward_inspect(self, clip_input, images):
    pooled_output = self.vision_model(clip_input)[1]
    image_embeds = self.visual_projection(pooled_output)

    special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds).cpu().numpy()
    cos_dist = cosine_distance(image_embeds, self.concept_embeds).cpu().numpy()

    matches = {"nsfw": [], "special": []}
    batch_size = image_embeds.shape[0]
    for i in range(batch_size):
        result_img = {
            "special_scores": {},
            "special_care": [],
            "concept_scores": {},
            "bad_concepts": [],
        }

        adjustment = -0.015

        for concet_idx in range(len(special_cos_dist[0])):
            concept_cos = special_cos_dist[i][concet_idx]
            concept_threshold = self.special_care_embeds_weights[concet_idx].item()
            result_img["special_scores"][concet_idx] = round(concept_cos - concept_threshold + adjustment, 3)
            if result_img["special_scores"][concet_idx] > 0:
                result_img["special_care"].append({concet_idx, result_img["special_scores"][concet_idx]})
                matches["special"].append(cst.SPECIAL_CONCEPTS[concet_idx])

        for concet_idx in range(len(cos_dist[0])):
            concept_cos = cos_dist[i][concet_idx]
            concept_threshold = self.concept_embeds_weights[concet_idx].item()
            result_img["concept_scores"][concet_idx] = round(concept_cos - concept_threshold + adjustment, 3)

            if result_img["concept_scores"][concet_idx] > 0:
                result_img["bad_concepts"].append(concet_idx)
                matches["nsfw"].append(cst.NSFW_CONCEPTS[concet_idx])

    has_nsfw_concepts = len(matches["nsfw"]) > 0

    return matches, has_nsfw_concepts


class Safety_Checker:
    def __init__(self):
        _device = os.getenv("DEVICE", cst.DEFAULT_DEVICE)
        self.device = _device if "cuda" in _device else f"cuda:{_device}"
        path = cst.SAFETY_CHECKER_REPO_PATH
        safety_pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.bfloat16).to(self.device)
        safety_pipe.safety_checker.forward = partial(forward_inspect, self=safety_pipe.safety_checker)
        self.safety_feature_extractor = safety_pipe.feature_extractor
        self.safety_checker = safety_pipe.safety_checker
        self.black_image = Image.open(cst.NSFW_IMAGE_PATH)

    def nsfw_check(self, image: Image.Image) -> Tuple[Image.Image, bool]:
        image_np = np.array(image)
        if np.all(image_np == 0):
            return True
        with torch.cuda.amp.autocast():
            safety_checker_input = self.safety_feature_extractor(images=image, return_tensors="pt").to(self.device)
            result, has_nsfw_concepts = self.safety_checker.forward(clip_input=safety_checker_input.pixel_values, images=image)
        return bool(has_nsfw_concepts)
