import bentoml
from pydantic import Field
import torch
import torchaudio
import base64
import os
from tempfile import NamedTemporaryFile
import numpy as np
import random
import soundfile as sf

from csm_custom.generator import Segment, load_csm_1b
import constants as cst

@bentoml.service()
class CSM:

    def __init__(self) -> None:
        self.generator = load_csm_1b(device="cuda")

    @bentoml.api
    async def generate(
        self,
        text_prompt_a: str = Field(
            description="Text caption for speaker A input audio",
            default=cst.DEFAULT_CAPTION_A
        ),
        audio_prompt_a: str = Field(
            description="Base64 encoded audio file for speaker A",
            default=cst.DEFAULT_REFERENCE_AUDIO_A
        ),
        text_prompt_b: str = Field(
            description="Text caption for speaker B input audio",
            default=cst.DEFAULT_CAPTION_B
        ),
        audio_prompt_b: str = Field(
            description="Base64 encoded audio file for speaker B", 
            default=cst.DEFAULT_REFERENCE_AUDIO_B
        ),
        conversation: str = Field(
            description="Conversation text with alternating lines between speakers",
            default="Hey how are you doing.\nPretty good, pretty good.\nI'm great, so happy to be speaking to you."
        ),
        max_audio_length_ms: int = Field(
            description="Maximum audio length in milliseconds",
            default=30000
        ),
        seed: int = Field(
            description="Random seed (currently not implemented)",
            default=0
        )
    ):
        
        random.seed(seed)
        np.random.seed(seed)
        # torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # Process audio prompts
        def decode_audio(base64_str: str) -> torch.Tensor:
            audio_bytes = base64.b64decode(base64_str.split(",")[-1])
            with NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_bytes)
                audio_tensor, sr = torchaudio.load(f.name)
                os.unlink(f.name)
            # Convert to mono (1D) if multi-channel
            if audio_tensor.shape[0] > 1:
                audio_tensor = torch.mean(audio_tensor, dim=0)
            else:
                audio_tensor = audio_tensor.squeeze(0)  # Ensure 1D
            # Resample if needed
            if sr != self.generator.sample_rate:
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor, orig_freq=sr, new_freq=self.generator.sample_rate
                )
            # Final check (optional but helpful during development)
            assert audio_tensor.ndim == 1, "Audio must be single channel"
            return audio_tensor

        speaker_a_audio = decode_audio(audio_prompt_a)
        speaker_b_audio = decode_audio(audio_prompt_b)

        # Prepare context segments
        context = [
            Segment(text=text_prompt_a, speaker=0, audio=speaker_a_audio),
            Segment(text=text_prompt_b, speaker=1, audio=speaker_b_audio)
        ]

        # Generate conversation
        generated = []
        lines = [line.strip() for line in conversation.split("\n") if line.strip()]
        
        for idx, line in enumerate(lines):
            speaker_id = idx % 2
            audio_tensor = self.generator.generate(
                text=line,
                speaker=speaker_id,
                context=context + generated,
                max_audio_length_ms=max_audio_length_ms
            )
            generated.append(Segment(text=line, speaker=speaker_id, audio=audio_tensor))

        full_audio = torch.cat([seg.audio for seg in generated], dim=0)

        # removed until profiling performance impact
        # watermarked_audio, wm_sr = watermark(
        #     self.generator._watermarker,
        #     full_audio,
        #     self.generator.sample_rate,
        #     self.watermark_key
        # )

        # Convert to byte format
        # audio_array = (watermarked_audio * 32768).to(torch.int16).cpu().numpy()

        audio_array = (full_audio * 32768).to(torch.int16).cpu().numpy()
        with NamedTemporaryFile() as f:
            sf.write(f.name, audio_array, self.generator.sample_rate, format="WAV")
            with open(f.name, "rb") as audio_file:
                base64_audio = base64.b64encode(audio_file.read()).decode("utf-8")

        return {
            "audio_b64": base64_audio
        }