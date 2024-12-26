import json
import constants as cst
from base_model import (
   InpaintingBase,
   UpscaleBase,
   TextToImageBase,
   ImageToImageBase, 
   AvatarBase,
   OutpaintingBase,
   LoadModelRequest
)
from typing import Dict, Any
from utils.base64_utils import base64_to_image
import os
from loguru import logger
from model_manager import model_manager
import copy
import random
import traceback


class PayloadModifier:
   def __init__(self):
       try:
           self._payloads = {}
           self.supported_workflows = []
           self._load_workflows()
           self.model_manager = model_manager
       except Exception as e:
           logger.error(f"Error in PayloadModifier initialization: {str(e)}\n{traceback.format_exc()}")
           raise

   def _load_workflows(self):
       try:
           directory = cst.WORKFLOWS_DIR
           for filename in os.listdir(directory):
               if filename.endswith(".json"):
                   self.supported_workflows.append(filename.split(".")[0])
                   filepath = os.path.join(directory, filename)
                   with open(filepath, "r") as file:
                       try:
                           data = json.load(file)
                           self._payloads[os.path.splitext(filename)[0]] = data
                       except json.JSONDecodeError as e:
                           logger.error(f"Error decoding JSON from {filename}: {e}\n{traceback.format_exc()}")
       except Exception as e:
           logger.error(f"Error loading workflows: {str(e)}\n{traceback.format_exc()}")
           raise

   def is_valid_model_workflow(self, model: str) -> bool:
       try:
           return model in self.supported_workflows
       except Exception as e:
           logger.error(f"Error checking model workflow validity: {str(e)}\n{traceback.format_exc()}")
           raise

   def is_valid_dynamic_string(self, model: str) -> bool:
       try:
           if "|" in model:
               repo_name, model_name = model.split("|", 1)
               return bool(model_name.strip()) and bool(repo_name.strip())
           return False
       except Exception as e:
           logger.error(f"Error checking dynamic string validity: {str(e)}\n{traceback.format_exc()}")
           raise

   def modify_inpaint(self, input_data: InpaintingBase) -> Dict[str, Any]:
       try:
           payload = copy.deepcopy(self._payloads["inpaint"])
           init_img = base64_to_image(input_data.init_image)
           init_img.save(f"{cst.COMFY_INPUT_PATH}init.png")
           mask_img = base64_to_image(input_data.mask_image)
           mask_img.save(f"{cst.COMFY_INPUT_PATH}mask.png")
           payload["Sampler"]["inputs"]["steps"] = input_data.steps
           payload["Sampler"]["inputs"]["cfg"] = input_data.cfg_scale
           positive_prompt, negative_prompt = input_data.prompt, input_data.negative_prompt
           payload["Prompt"]["inputs"]["text"] = positive_prompt
           payload["Negative_prompt"]["inputs"]["text"] += negative_prompt
           seed = input_data.seed or random.randint(1, 2**16)
           payload["Sampler"]["inputs"]["noise_seed"] = seed
           return payload
       except Exception as e:
           logger.error(f"Error in modify_inpaint: {str(e)}\n{traceback.format_exc()}")
           raise

   def modify_outpaint(self, input_data: OutpaintingBase) -> Dict[str, Any]:
       try:
           payload = copy.deepcopy(self._payloads["outpaint"])
           init_img = base64_to_image(input_data.init_image)
           init_img.save(f"{cst.COMFY_INPUT_PATH}init.png")
           positive_prompt, negative_prompt = input_data.prompt, input_data.negative_prompt
           payload["Prompt"]["inputs"]["text"] = positive_prompt
           payload["Negative_prompt"]["inputs"]["text"] += negative_prompt
           for position in input_data.pad_values:
               payload["Outpaint_pad"]["inputs"][position] = input_data.pad_values[position]
           seed = input_data.seed or random.randint(1, 2**16)
           payload["Sampler"]["inputs"]["noise_seed"] = seed
           return payload
       except Exception as e:
           logger.error(f"Error in modify_outpaint: {str(e)}\n{traceback.format_exc()}")
           raise

   async def modify_text_to_image(self, input_data: TextToImageBase) -> Dict[str, Any]:
       try:
           if self.is_valid_model_workflow(input_data.model.strip().lower()):
               try:
                   payload = copy.deepcopy(self._payloads[f"{input_data.model}"])
                   positive_prompt, negative_prompt = input_data.prompt, input_data.negative_prompt
                   payload["Prompt"]["inputs"]["text"] = positive_prompt
                   payload["Sampler"]["inputs"]["steps"] = input_data.steps
                   payload["Latent"]["inputs"]["width"] = input_data.width
                   payload["Latent"]["inputs"]["height"] = input_data.height
                   seed = input_data.seed or random.randint(1, 2**16)
                   if "flux" in input_data.model:
                       payload["Seed"]["inputs"]["noise_seed"] = seed
                       payload["Guidance"]["inputs"]["guidance"] = input_data.cfg_scale
                   else:
                       payload["Negative_prompt"]["inputs"]["text"] += negative_prompt
                       payload["Sampler"]["inputs"]["cfg"] = input_data.cfg_scale
                       payload["Sampler"]["inputs"]["seed"] = seed
                   return payload
               except Exception as e:
                   logger.error(f"Error processing valid model workflow: {str(e)}\n{traceback.format_exc()}")
                   raise
           else:
               try:
                   if self.is_valid_dynamic_string(input_data.model):
                       model_repo, safetensors_filename = [val.strip() for val in input_data.model.split('|')]
                       logger.info(f"Loading {safetensors_filename}")
                       load_model_data = LoadModelRequest(model_repo=model_repo, safetensors_filename=safetensors_filename)
                       load_model_response = await self.model_manager.download_model(load_model_data)
                       input_data.model = safetensors_filename
                       logger.info(load_model_response.status)
                   else:
                       input_data.model = self.model_manager.get_last_used_model()
                       logger.info("Using last loaded model")
                   payload = json.dumps(copy.deepcopy(self._payloads["dynamic-text-to-image"]))
                   if input_data.seed == 0:
                       input_data.seed = random.randint(1, 2**16)
                   for key, value in input_data:
                       placeholder = f"{{{{{key}}}}}"
                       replacement_value = str(value) if not isinstance(value, str) else value
                       payload = payload.replace(placeholder, replacement_value)
                   return json.loads(payload)
               except Exception as e:
                   logger.error(f"Error processing dynamic model: {str(e)}\n{traceback.format_exc()}")
                   raise
       except Exception as e:
           logger.error(f"Error in modify_text_to_image: {str(e)}\n{traceback.format_exc()}")
           raise

   def modify_image_to_image(self, input_data: ImageToImageBase) -> Dict[str, Any]:
       try:
           payload = copy.deepcopy(self._payloads[f"{input_data.model}"])
           init_img = base64_to_image(input_data.init_image)
           init_img.save(f"{cst.COMFY_INPUT_PATH}init.png")
           positive_prompt, negative_prompt = input_data.prompt, input_data.negative_prompt
           payload["Prompt"]["inputs"]["text"] = positive_prompt
           payload["Sampler"]["inputs"]["steps"] = input_data.steps
           payload["Sampler"]["inputs"]["denoise"] = 1 - input_data.image_strength
           seed = input_data.seed or random.randint(1, 2**16)
           if "flux" in input_data.model:
               payload["Seed"]["inputs"]["noise_seed"] = seed
               payload["Guidance"]["inputs"]["guidance"] = input_data.cfg_scale
           else:
               payload["Negative_prompt"]["inputs"]["text"] += negative_prompt
               payload["Sampler"]["inputs"]["cfg"] = input_data.cfg_scale
               payload["Sampler"]["inputs"]["seed"] = seed
           logger.debug(f"payload: {payload}")
           return payload
       except Exception as e:
           logger.error(f"Error in modify_image_to_image: {str(e)}\n{traceback.format_exc()}")
           raise

   def modify_upscale(self, input_data: UpscaleBase) -> Dict[str, Any]:
       try:
           workflow_name = "upscale_sampled" if input_data.sampled else "upscale"
           payload = copy.deepcopy(self._payloads[workflow_name])
           init_img = base64_to_image(input_data.init_image)
           init_img.save(f"{cst.COMFY_INPUT_PATH}init.png")
           return payload
       except Exception as e:
           logger.error(f"Error in modify_upscale: {str(e)}\n{traceback.format_exc()}")
           raise

   def modify_avatar(self, input_data: AvatarBase) -> Dict[str, Any]:
       try:
           logger.info("ZZZ")
           payload = copy.deepcopy(self._payloads["instantid"])
           init_img = base64_to_image(input_data.init_image)
           init_img.save(f"{cst.COMFY_INPUT_PATH}init.png")
           positive_prompt, negative_prompt = input_data.prompt, input_data.negative_prompt
           payload["Prompt"]["inputs"]["text"] += positive_prompt
           payload["Negative_prompt"]["inputs"]["text"] += negative_prompt
           payload["Sampler"]["inputs"]["steps"] = input_data.steps
           seed = input_data.seed or random.randint(1, 2**16)
           payload["Sampler"]["inputs"]["seed"] = seed
           payload["Sampler_initial"]["inputs"]["seed"] = seed
           payload["Latent"]["inputs"]["width"] = input_data.width
           payload["Latent"]["inputs"]["height"] = input_data.height
           payload["InstantID"]["inputs"]["ip_weight"] = input_data.ipadapter_strength
           payload["InstantID"]["inputs"]["cn_strength"] = input_data.control_strength
           logger.info("ZZZ - 2")
           return payload
       except Exception as e:
           logger.error(f"Error in modify_avatar: {str(e)}\n{traceback.format_exc()}")
           raise