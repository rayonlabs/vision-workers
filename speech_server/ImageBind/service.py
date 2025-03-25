import bentoml

from typing import Optional, Dict, List
import data
import torch
from PIL import Image
import librosa
import numpy as np
import base64
import io
import requests
from models import imagebind_model
from models.imagebind_model import ModalityType

from pydantic import Field

MODALITY_TO_PREPROCESSING = {
    ModalityType.TEXT: data.load_and_transform_text,
    ModalityType.VISION: data.load_and_transform_vision_data,
    ModalityType.AUDIO: data.load_and_transform_audio_data,
}



@bentoml.service()
class ImageBind:
    
    def __init__(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        
        model = imagebind_model.imagebind_huge(pretrained=True)
        model.eval()
        self.model = model.to("cuda")
        self.model({ModalityType.TEXT: MODALITY_TO_PREPROCESSING[ModalityType.TEXT](["The quick brown fox ... uhhh .. idk"], "cuda")})
        torch.backends.cuda.cufft_plan_cache[0].max_size = 0

    @bentoml.api
    async def predictions(self,
        input1: str = Field(
            None,
            description="Input 1. Can be of modality: vision, audio, or a web link to an image or audio file."
        ),
        input2: str = Field(
            None,
            description="Input 2. Can be of modality: vision, audio, or a web link to an image or audio file."
        ),
        text_input: str = Field(
            None,
            description="Optional text input that can be embedded."
        )
        ):
        """Calculate cosine similarity between two inputs or one input and text"""

        if not input1 and not input2 and not text_input:
            raise Exception("At least one input must be provided!")

        device = "cuda"
        embeddings = []

        # Process input1 or input2 if provided
        for input_data in [input1, input2]:
            if input_data is not None:
                modality = self.recognize_modality(input_data)
                modality_function = MODALITY_TO_PREPROCESSING[modality]
                
                if modality != ModalityType.TEXT:
                    input_file = self.load_data_from_url_or_base64(input_data)
                    model_input = {modality: modality_function([input_file], device)}
                else:
                    model_input = {modality: modality_function([input_data], device)}
                
                with torch.no_grad():
                    embedding = self.model(model_input)
                embeddings.append(embedding[modality].squeeze())

        # Process text_input if provided
        if text_input is not None:
            model_input = {ModalityType.TEXT: MODALITY_TO_PREPROCESSING[ModalityType.TEXT]([text_input], device)}
            with torch.no_grad():
                embedding = self.model(model_input)
            embeddings.append(embedding[ModalityType.TEXT].squeeze())

        print("Embeddings:", embeddings)
        # Calculate cosine similarity if two embeddings are provided
        if len(embeddings) == 2:
            similarity = self.cosine_similarity(embeddings[0], embeddings[1])
            return {'output': [similarity]}
        # Return the single embedding if only one input is provided
        elif len(embeddings) == 1:
            return {'output': embeddings[0].tolist()}
        else:
            raise Exception("Unexpected number of inputs provided!")
    
        
    @staticmethod
    def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        sim = torch.nn.CosineSimilarity(dim=0)(emb1, emb2)
        return sim.item()

    
    @staticmethod
    def recognize_modality(input_data: str) -> ModalityType:
        if input_data.startswith("http") and any(ext in input_data for ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]):
            return ModalityType.VISION
        elif input_data.startswith("http") and any(ext in input_data for ext in [".wav", ".mp3", ".ogg", ".flac"]):
            return ModalityType.AUDIO
        elif input_data.startswith("data:image"):
            return ModalityType.VISION
        elif input_data.startswith("data:audio"):
            return ModalityType.AUDIO
        else:
            return ModalityType.TEXT
        
    @staticmethod
    def load_data_from_url_or_base64(input_data: str) -> io.BytesIO:
        if input_data.startswith("http"):
            response = requests.get(input_data)
            return io.BytesIO(response.content)
        elif input_data.startswith("data:"):
            _, encoded_data = input_data.split(",", 1)
            return io.BytesIO(base64.b64decode(encoded_data))
        else:
            raise ValueError("Unsupported input format")