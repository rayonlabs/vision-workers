import json
import constants as cst
from base_model import (
    ImagebindEmbeddingsBase
)
from typing import Dict, Any, Tuple, List
# from utils.base64_utils import base64_to_image
import os
import copy
import random


class PayloadModifier:
    def __init__(self):
        self._payloads = {}