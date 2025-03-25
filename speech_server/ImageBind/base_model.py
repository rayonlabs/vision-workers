from typing import Dict, List, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
import constants as cst
# from 'styletts2'.service import LanguageEnum, StyleTTS2ModelEnum


class ImagebindEmbeddingsBase(BaseModel):
    input1: Optional[str] = Field(
        cst.DEFAULT_REFERENCE_AUDIO,
        description="The first audio b64",
        title="input1"
    )

    input2: Optional[str] = Field(
        None,
        description="The second audio b64",
        title="input2",
    )

    text_input: Optional[str] = Field(
        None,
        description="The raw text to be embedded either standalone or with the audio (cosine similarity)",
        title="text_input",
    )

#ImageBind either returns (i) cosine similarity score between two embeddings or (ii) embeddings of a single input
class ImagebindEmbeddingsResponse(BaseModel):
    imagebind_embeddings: Optional[List[float]] =  Field(None, description="The imagebind embeddings", title="imagebind_embeddings", alias = "output")

    class Config:
        populate_by_name = True  # Allow both aliases and field names
        allow_population_by_field_name = True  # Allow both aliases and field names