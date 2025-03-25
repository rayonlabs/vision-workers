import enum
from typing import List, Optional
from pydantic import BaseModel


class Role(str, enum.Enum):
    """Message is sent by which role?"""

    user = "user"
    assistant = "assistant"
    system = "system"


class Message(BaseModel):
    role: Role = Role.user
    content: str = "Remind me that I have forgot to set the messages"

    class Config:
        extra = "allow"


class TextPrompt(BaseModel):
    text: str
    weight: Optional[float]


class ImageHashes(BaseModel):
    average_hash: str = ""
    perceptual_hash: str = ""
    difference_hash: str = ""
    color_hash: str = ""


class ImageResponseBody(BaseModel):
    image_b64: Optional[str] = None
    is_nsfw: Optional[bool] = None
    clip_embeddings: Optional[List[float]] = None
    image_hashes: Optional[ImageHashes] = None

class ClipEmbeddingsResponse(BaseModel):
    clip_embeddings: Optional[List[List[float]]] = None

class CSMResponseBody(BaseModel):
    audio_b64: Optional[str] = None