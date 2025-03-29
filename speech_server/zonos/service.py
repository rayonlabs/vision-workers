import bentoml
from pydantic import Field
import torch
import torchaudio
import base64
import os
from tempfile import NamedTemporaryFile
import numpy as np
import soundfile as sf

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict, supported_language_codes
from constants import DEFAULT_REFERENCE_AUDIO_A

@bentoml.service()
class ZonosTTS:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {
            "Zyphra/Zonos-v0.1-transformer": Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=self.device),
            "Zyphra/Zonos-v0.1-hybrid": Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=self.device)
        }
        for model in self.models.values():
            model.requires_grad_(False).eval()

    @bentoml.api
    async def generate(
        self,
        model_choice: str = Field(
            description="Model variant to use",
            default="Zyphra/Zonos-v0.1-transformer",
            enum=["Zyphra/Zonos-v0.1-transformer", "Zyphra/Zonos-v0.1-hybrid"]
        ),
        text: str = Field(
            description="Text to synthesize",
            default="Zonos uses eSpeak for text to phoneme conversion!"
        ),
        language: str = Field(
            description="Language code for synthesis",
            default="en-us",
            enum=supported_language_codes
        ),
        speaker_audio: str = Field(
            description="Base64 encoded audio file for speaker reference (optional)",
            default=DEFAULT_REFERENCE_AUDIO_A
        ),
        prefix_audio: str = Field(
            description="Base64 encoded audio file to continue from (optional)",
            default=None
        ),
        emotions: list[float] = Field(
            description="List of 8 emotion values between 0.0 and 1.0",
            default=[1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2]
        ),
        vq_single: float = Field(
            description="VQ score value between 0.5 and 0.8",
            default=0.78
        ),
        fmax: float = Field(
            description="Maximum frequency in Hz",
            default=24000.0
        ),
        pitch_std: float = Field(
            description="Pitch standard deviation",
            default=45.0
        ),
        speaking_rate: float = Field(
            description="Speaking rate",
            default=15.0
        ),
        dnsmos_ovrl: float = Field(
            description="DNSMOS overall score target",
            default=4.0
        ),
        speaker_noised: bool = Field(
            description="Whether to denoise speaker",
            default=True
        ),
        cfg_scale: float = Field(
            description="CFG scale for generation",
            default=2.0
        ),
        min_p: float = Field(
            description="Min P sampling parameter",
            default=0.15
        ),
        seed: int = Field(
            description="Random seed (use -1 for random)",
            default=-1
        ),
        unconditional_keys: list[str] = Field(
            description="List of conditioning keys to make unconditional",
            default=["emotion"]
        )
    ) -> dict:
        # Select model
        selected_model = self.models[model_choice]
        
        # Handle seed
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        torch.manual_seed(seed)
        
        # Process speaker audio if provided
        speaker_embedding = None
        if speaker_audio and "speaker" not in unconditional_keys:
            wav, sr = self._load_audio_from_base64(speaker_audio)
            speaker_embedding = selected_model.make_speaker_embedding(wav, sr)
            speaker_embedding = speaker_embedding.to(self.device, dtype=torch.bfloat16)
        
        # Process prefix audio if provided
        audio_prefix_codes = None
        if prefix_audio:
            wav_prefix, sr_prefix = self._load_audio_from_base64(prefix_audio)
            wav_prefix = wav_prefix.mean(0, keepdim=True)
            wav_prefix = torchaudio.functional.resample(
                wav_prefix, 
                sr_prefix, 
                selected_model.autoencoder.sampling_rate
            )
            wav_prefix = wav_prefix.to(self.device, dtype=torch.float32)
            with torch.autocast(self.device, dtype=torch.float32):
                audio_prefix_codes = selected_model.autoencoder.encode(wav_prefix.unsqueeze(0))
        
        # Prepare emotion tensor
        emotion_tensor = torch.tensor(emotions, device=self.device)
        
        # Prepare VQ tensor
        vq_tensor = torch.tensor([vq_single] * 8, device=self.device).unsqueeze(0)
        
        # Create conditioning dictionary
        cond_dict = make_cond_dict(
            text=text,
            language=language,
            speaker=speaker_embedding,
            emotion=emotion_tensor,
            vqscore_8=vq_tensor,
            fmax=fmax,
            pitch_std=pitch_std,
            speaking_rate=speaking_rate,
            dnsmos_ovrl=dnsmos_ovrl,
            speaker_noised=speaker_noised,
            device=self.device,
            unconditional_keys=unconditional_keys,
        )
        conditioning = selected_model.prepare_conditioning(cond_dict)
        
        # Generate audio
        max_new_tokens = 86 * 30  # Approx 30 seconds
        codes = selected_model.generate(
            prefix_conditioning=conditioning,
            audio_prefix_codes=audio_prefix_codes,
            max_new_tokens=max_new_tokens,
            cfg_scale=cfg_scale,
            batch_size=1,
            sampling_params={"min_p": min_p},
        )
        
        # Decode and process output
        wav_out = selected_model.autoencoder.decode(codes).cpu().detach()
        sr_out = selected_model.autoencoder.sampling_rate
        if wav_out.dim() == 2 and wav_out.size(0) > 1:
            wav_out = wav_out[0:1, :]
        
        # Convert to base64
        audio_array = (wav_out.squeeze().numpy() * 32768).astype(np.int16)
        with NamedTemporaryFile() as f:
            sf.write(f.name, audio_array, sr_out, format="WAV")
            with open(f.name, "rb") as audio_file:
                base64_audio = base64.b64encode(audio_file.read()).decode("utf-8")
        
        return {
            "audio_b64": base64_audio,
        }
    
    def _load_audio_from_base64(self, base64_str: str) -> tuple[torch.Tensor, int]:
        """Helper to load audio from base64 string"""
        audio_bytes = base64.b64decode(base64_str.split(",")[-1])
        with NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            audio_tensor, sr = torchaudio.load(f.name)
            os.unlink(f.name)
        return audio_tensor, sr