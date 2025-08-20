import numpy as np
from PIL import Image
from functools import partial
import torch
from torch import nn
from diffusers import StableDiffusionPipeline
from typing import Tuple
import constants as cst
import os
import threading


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


@torch.no_grad()
def forward_inspect(self, clip_input, images):
    pooled_output = self.vision_model(clip_input)[1]
    image_embeds = self.visual_projection(pooled_output)

    special_cos = cosine_distance(image_embeds, self.special_care_embeds).cpu().numpy()
    concept_cos = cosine_distance(image_embeds, self.concept_embeds).cpu().numpy()

    special_thr = self.special_care_embeds_weights.detach().cpu().numpy()
    concept_thr = self.concept_embeds_weights.detach().cpu().numpy()

    adjustment = -0.015

    raw_scores = []
    matches = {"nsfw": [], "special": []}

    batch_size = image_embeds.shape[0]
    for i in range(batch_size):
        result_img = {
            "special_scores": special_cos[i].tolist(),
            "concept_scores": concept_cos[i].tolist(),
        }
        raw_scores.append(result_img)

        for idx, sim in enumerate(special_cos[i]):
            score = sim - special_thr[idx] + adjustment
            if score > 0:
                matches["special"].append((idx, round(float(score), 3)))

        for idx, sim in enumerate(concept_cos[i]):
            score = sim - concept_thr[idx] + adjustment
            if score > 0:
                matches["nsfw"].append((idx, round(float(score), 3)))

    has_nsfw_concepts = len(matches["nsfw"]) > 0

    return raw_scores, has_nsfw_concepts



class Safety_Checker:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Safety_Checker, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if Safety_Checker._initialized:
            return
            
        with Safety_Checker._lock:
            if Safety_Checker._initialized:
                return
                
            print("Loading Safety Checker model (this will only happen once)...")
            _device = os.getenv("DEVICE", cst.DEFAULT_DEVICE)
            self.device = _device if "cuda" in _device else f"cuda:{_device}"
            path = cst.SAFETY_CHECKER_REPO_PATH
            safety_pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.bfloat16).to(self.device)
            safety_pipe.safety_checker.forward = partial(forward_inspect, self=safety_pipe.safety_checker)
            self.safety_feature_extractor = safety_pipe.feature_extractor
            self.safety_checker = safety_pipe.safety_checker
            self.black_image = Image.open(cst.NSFW_IMAGE_PATH)
            
            Safety_Checker._initialized = True
            print("Safety Checker model loaded successfully!")

    def get_nsfw_scores(self, image: Image.Image) -> Tuple[Image.Image, bool]:
        image_np = np.array(image)
        if np.all(image_np == 0):
            return True
        with torch.cuda.amp.autocast():
            safety_checker_input = self.safety_feature_extractor(images=image, return_tensors="pt").to(self.device)
            scores, is_nsfw = self.safety_checker.forward(clip_input=safety_checker_input.pixel_values, images=image)
        return scores, is_nsfw