from typing import Dict, List, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
import constants as cst


class ModelEnum(str, Enum):
    DREAMSHAPER = "dreamshaper"
    PROTEUS = "proteus"
    PLAYGROUND = "playground"
    FLUX_SCHNELL = "flux"
    FLUX_KONTEXT = "flux-kontext"


class SamplerEnum(str, Enum):
    EULER = "euler"
    EULER_CFG_PP = "euler_cfg_pp"
    EULER_ANCESTRAL = "euler_ancestral"
    EULER_ANCESTRAL_CFG_PP = "euler_ancestral_cfg_pp"
    HEUN = "heun"
    HEUNPP2 = "heunp2"
    DPM_2 = "dpm_2"
    DPM_2_ANCESTRAL = "dpm_2_ancestral"
    LMS = "lms"
    DPM_FAST = "dpm_fast"
    DPM_ADAPTIVE = "dpm_adaptive"
    DPMPP_2S_ANCESTRAL = "dpmpp_2s_ancestral"
    DPMPP_SDE_GPU = "dpmpp_sde_gpu"
    DPMPP_2M_SDE_GPU = "dpmpp_2m_sde_gpu"
    DPMPP_3M_SDE_GPU = "dpmpp_3m_sde_gpu"
    DDPM = "ddpm"
    LCM = "lcm"
    IPNDM = "ipndm"
    PNDM = "pndm"
    UNI_PC = "uni_pc"
    UNI_PC_BH2 = "uni_pc_bh2"


class SchedulerEnum(str, Enum):
    NORMAL = "normal"
    KARRAS = "karras"
    EXPONENTIAL = "exponential"
    SGM_UNIFORM = "sgm_uniform"
    SIMPLE = "simple"
    DDIM_UNIFORM = "ddim_uniform"
    BETA = "beta"


class ModelStatus(str, Enum):
    ALREADY_EXISTS = "Model already exists"
    SUCCESS = "Model downloaded successfully"


class KontextLoraEnum(str, Enum):
    PUT_IT_HERE = "put_it_here"
    JPEG_COMP_FIX = "jpeg_comp_fix"
    INSCENE = "inscene"


class LoadModelRequest(BaseModel):
    model_repo: str = Field(..., example="Lykon/dreamshaper-xl-lightning")
    safetensors_filename: str = Field(..., example="DreamShaperXL_Lightning-SFW.safetensors")


class LoadModelResponse(BaseModel):
    status: ModelStatus


class TextToImageBase(BaseModel):
    prompt: str = Field(..., description="The prompt to generate the image")
    negative_prompt: str = Field(default="", description="The negative prompt to generate the image")
    steps: int = Field(..., description="Number of inference steps, higher for more quality but increased generation time", gt=4, lt=50)
    model: str = Field(..., description="The engine to use for image generation")
    cfg_scale: float = Field(..., description="Guidance scale", gt=1.5, lt=12)
    height: int = Field(..., description="Height of the output image in pixels", ge=512, lt=2048)
    width: int = Field(..., description="Width of the output image in pixels", ge=512, lt=2048)
    seed: int = Field(..., description="Seed value for deterministic outputs", ge=0)
    sampler: SamplerEnum = Field(default=SamplerEnum.DPMPP_SDE_GPU, description="The sampling method to use during image generation")
    scheduler: SchedulerEnum = Field(default=SchedulerEnum.KARRAS, description="The scheduler to use for adjusting noise and guidance during generation")


class ImageToImageBase(BaseModel):
    prompt: str = Field(..., description="The prompt to generate the image")
    negative_prompt: str = Field(default="", description="The negative prompt to generate the image")
    init_image: str
    model: str = Field(..., description="The engine to use for image generation")
    image_strength: float = Field(..., description="Image strength of the generated image with respect to the original image", gt=0.01, lt=1)
    steps: int = Field(..., description="Number of inference steps, higher for more quality but increased generation time", gt=4, lt=50)
    cfg_scale: float = Field(..., description="Guidance scale", gt=1, lt=12)
    seed: int = Field(..., description="Seed value for deterministic outputs", ge=0)


class ImageEditBase(BaseModel):
    prompt: str = Field(..., description="The prompt to generate the image")
    init_image: str
    steps: int = Field(..., description="Number of inference steps, higher for more quality but increased generation time", gt=4, lt=50)
    cfg_scale: float = Field(..., description="Guidance scale", gt=1, lt=12)
    seed: int = Field(..., description="Seed value for deterministic outputs", ge=0)
    height: int = Field(default=1024, description="Height of the output image in pixels (optional, between 512 and 2048)", ge=512, lt=2048)
    width: int = Field(default=1024, description="Width of the output image in pixels (optional, between 512 and 2048)", ge=512, lt=2048)
    model: str = Field(default="flux-kontext-image-edit", title="Model")
    lora: KontextLoraEnum = Field(default=None, description="Optional LoRA to apply to kontext")


class UpscaleBase(BaseModel):
    init_image: str
    sampled: bool = Field(default=True)


class AvatarBase(BaseModel):
    prompt: str = Field(..., description="The prompt to generate the image")
    negative_prompt: str = Field(default="", description="The negative prompt to generate the image")
    init_image: str
    ipadapter_strength: float = Field(..., description="IP Adapter strength, increase for more face coherence, works best on default", gt=0.1, le=1)
    control_strength: float = Field(..., description="Control strength, increase for more face coherence, works best on default", gt=0.1, le=1.01)
    steps: int = Field(..., description="Number of inference steps, higher for more quality but increased generation time", gt=4, lt=50)
    height: int = Field(..., description="Height of the output image in pixels", ge=512, lt=2048)
    width: int = Field(..., description="Width of the output image in pixels", ge=512, lt=2048)
    seed: int = Field(..., description="Seed value for deterministic outputs", ge=0)


class InpaintingBase(BaseModel):
    prompt: str = Field(..., description="The prompt to generate the image")
    negative_prompt: str = Field(default="", description="The negative prompt to generate the image")
    init_image: str
    mask_image: str
    steps: int = Field(
        cst.DEFAULT_STEPS_INPAINT,
        description="Number of inference steps, higher for more quality but increased generation time",
        gt=4,
        lt=50,
    )
    cfg_scale: float = Field(cst.DEFAULT_CFG_INPAINT, description="Guidance scale", gt=1.5, lt=12)
    seed: int = Field(..., description="Seed value for deterministic outputs", ge=0)


class OutpaintingBase(BaseModel):
    text_prompts: List[Dict[str, Any]]
    init_image: str
    text_prompts: List[Dict[str, Any]] = Field(default=[{"text": "", "weight": 1}], description="Text prompts to guide the generation process")
    pad_values: dict = Field(
        default={"left": 104, "right": 104, "top": 104, "bottom": 104},
        description="Dictionary specifying padding in pixels for each side of the image for expansion. Format: {'left': int, 'right': int, 'top': int, 'bottom': int}",
    )
    steps: int = Field(
        cst.DEFAULT_STEPS_INPAINT,
        description="Number of inference steps, higher for more quality but increased generation time",
        gt=4,
        lt=50,
    )
    cfg_scale: float = Field(cst.DEFAULT_CFG_INPAINT, description="Guidance scale", gt=1.5, lt=12)
    seed: int = Field(..., description="Seed value for deterministic outputs", ge=0)


class ClipEmbeddingsBase(BaseModel):
    image_b64s: Optional[List[str]] = Field(
        None,
        description="The image b64s",
        title="image_b64s",
    )


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


class ClipEmbeddingsTextBase(BaseModel):
    text_prompt: str


class ClipEmbeddingsTextResponse(BaseModel):
    text_embedding: Optional[List[float]] = None


class CheckNSFWBase(BaseModel):
    image: str


class CheckNSFWResponse(BaseModel):
    is_nsfw: bool
