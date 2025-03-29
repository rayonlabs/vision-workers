from typing import Dict, List, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
import constants as cst
from zonos.conditioning import supported_language_codes
class TextToSpeechBase(BaseModel):
    model_choice: str = Field(
        description="Model variant to use",
        default="Zyphra/Zonos-v0.1-transformer",
        enum=["Zyphra/Zonos-v0.1-transformer", "Zyphra/Zonos-v0.1-hybrid"]
    )
    text: str = Field(
        description="Text to synthesize",
        default="Zonos uses eSpeak for text to phoneme conversion!"
    )
    language: str = Field(
        description="Language code for synthesis",
        default="en-us",
        enum=supported_language_codes
    )
    speaker_audio: str = Field(
        description="Base64 encoded audio file for speaker reference (optional)",
        default=cst.DEFAULT_REFERENCE_AUDIO_A
    )
    prefix_audio: str = Field(
        description="Base64 encoded audio file to continue from (optional)",
        default=None
    )
    emotions: list[float] = Field(
        description="List of 8 emotion values between 0.0 and 1.0",
        default=[1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2]
    )
    vq_single: float = Field(
        description="VQ score value between 0.5 and 0.8",
        default=0.78
    )
    fmax: float = Field(
        description="Maximum frequency in Hz",
        default=24000.0
    )
    pitch_std: float = Field(
        description="Pitch standard deviation",
        default=45.0
    )
    speaking_rate: float = Field(
        description="Speaking rate",
        default=15.0
    )
    dnsmos_ovrl: float = Field(
        description="DNSMOS overall score target",
        default=4.0
    )
    speaker_noised: bool = Field(
        description="Whether to denoise speaker",
        default=True
    )
    cfg_scale: float = Field(
        description="CFG scale for generation",
        default=2.0
    )
    min_p: float = Field(
        description="Min P sampling parameter",
        default=0.15
    )
    seed: int = Field(
        description="Random seed (use -1 for random)",
        default=-1
    )
    unconditional_keys: list[str] = Field(
        description="List of conditioning keys to make unconditional",
        default=["emotion"]
    )

class ZonosIncoming(TextToSpeechBase):
    ...

class ZonosResponseBody(BaseModel):
    audio_b64: Any = Field(None, description="The synthesised base64 audio", title="audio_b64", alias = "audio_b64")

    class Config:
        populate_by_name = True  # Allow both aliases and field names
        allow_population_by_field_name = True  # Allow both aliases and field names