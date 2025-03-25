from typing import Dict, List, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
import constants as cst


class TextToSpeechBase(BaseModel):
    text_prompt_a: str = Field(
            description="Text caption for speaker A input audio",
            default=cst.DEFAULT_CAPTION_A
    )
    audio_prompt_a: str = Field(
        description="Base64 encoded audio file for speaker A",
        default=cst.DEFAULT_REFERENCE_AUDIO_A
    )
    text_prompt_b: str = Field(
        description="Text caption for speaker B input audio",
        default=cst.DEFAULT_CAPTION_B
    )
    audio_prompt_b: str = Field(
        description="Base64 encoded audio file for speaker B", 
        default=cst.DEFAULT_REFERENCE_AUDIO_B
    )
    conversation: str = Field(
        description="Conversation text with alternating lines between speakers",
        default="Hey how are you doing.\nPretty good, pretty good.\nI'm great, so happy to be speaking to you."
    )
    max_audio_length_ms: int = Field(
        description="Maximum audio length in milliseconds",
        default=30000
    )
    seed: int = Field(
        description="Random seed (currently not implemented)",
        default=0
    )

class CSMIncoming(TextToSpeechBase):
    ...

class CSMResponseBody(BaseModel):
    audio_b64: Any = Field(None, description="The synthesised base64 audio", title="audio_b64", alias = "audio_b64")

    class Config:
        populate_by_name = True  # Allow both aliases and field names
        allow_population_by_field_name = True  # Allow both aliases and field names