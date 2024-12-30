"""
Task config. get this from  https://github.com/namoray/nineteen/blob/production/core/task_config.py
Just add `sdk.` to the imports
"""

from sdk.logging import get_logger
from sdk.core.models import config_models as cmodels

logger = get_logger(__name__)


CHAT_LLAMA_3_2_3B = "chat-llama-3-2-3b"
CHAT_LLAMA_3_1_70B = "chat-llama-3-1-70b"
CHAT_LLAMA_3_1_8B = "chat-llama-3-1-8b"
CHAT_ROGUE_ROSE_103B = "chat-rogue-rose-103b"
PROTEUS_TEXT_TO_IMAGE = "proteus-text-to-image"
PROTEUS_IMAGE_TO_IMAGE = "proteus-image-to-image"
FLUX_SCHNELL_TEXT_TO_IMAGE = "flux-schnell-text-to-image"
FLUX_SCHNELL_IMAGE_TO_IMAGE = "flux-schnell-image-to-image"
AVATAR = "avatar"
DREAMSHAPER_TEXT_TO_IMAGE = "dreamshaper-text-to-image"
DREAMSHAPER_IMAGE_TO_IMAGE = "dreamshaper-image-to-image"


def task_configs_factory() -> dict[str, cmodels.FullTaskConfig]:
    return {
        CHAT_LLAMA_3_2_3B: cmodels.FullTaskConfig(
            task=CHAT_LLAMA_3_2_3B,
            task_type=cmodels.TaskType.TEXT,
            max_capacity=120_000,
            orchestrator_server_config=cmodels.OrchestratorServerConfig(
                server_needed=cmodels.ServerType.LLM,
                load_model_config={
                    "model": "unsloth/Llama-3.2-3B-Instruct",
                    "half_precision": True,
                    "tokenizer": "tau-vision/llama-tokenizer-fix",
                    "max_model_len": 20_000,
                    "gpu_memory_utilization": 0.65,
                    "eos_token_id": 128009,
                },
                endpoint=cmodels.Endpoints.chat_completions.value,
                checking_function="check_text_result",
                task=CHAT_LLAMA_3_2_3B,
            ),
            synthetic_generation_config=cmodels.SyntheticGenerationConfig(func="generate_chat_synthetic", kwargs={"model": CHAT_LLAMA_3_2_3B}),
            endpoint=cmodels.Endpoints.chat_completions.value,
            volume_to_requests_conversion=300,
            is_stream=True,
            weight=0.05,
            timeout=2,
            enabled=True,
        ),
        CHAT_ROGUE_ROSE_103B: cmodels.FullTaskConfig(
            task=CHAT_ROGUE_ROSE_103B,
            task_type=cmodels.TaskType.TEXT,
            max_capacity=120_000,
            orchestrator_server_config=cmodels.OrchestratorServerConfig(
                server_needed=cmodels.ServerType.LLM,
                load_model_config={
                    "model": "sophosympatheia/Rogue-Rose-103b-v0.2",
                    "tokenizer": "sophosympatheia/Rogue-Rose-103b-v0.2",
                    "revision": "exl2-3.2bpw",
                    "half_precision": True,
                    "max_model_len": 4096,
                    "gpu_memory_utilization": 0.7,
                    "eos_token_id": 2,
                },
                endpoint=cmodels.Endpoints.chat_completions.value,
                task=CHAT_ROGUE_ROSE_103B,
                checking_function="check_text_result",
            ),
            synthetic_generation_config=cmodels.SyntheticGenerationConfig(func="generate_chat_synthetic", kwargs={"model": CHAT_ROGUE_ROSE_103B}),
            endpoint=cmodels.Endpoints.chat_completions.value,
            volume_to_requests_conversion=300,
            is_stream=True,
            weight=0.05,
            timeout=2,
            enabled=True,
        ),
        CHAT_LLAMA_3_1_70B: cmodels.FullTaskConfig(
            task=CHAT_LLAMA_3_1_70B,
            task_type=cmodels.TaskType.TEXT,
            max_capacity=120_000,
            orchestrator_server_config=cmodels.OrchestratorServerConfig(
                server_needed=cmodels.ServerType.LLM,
                load_model_config={
                    "model": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
                    "half_precision": True,
                    "tokenizer": "tau-vision/llama-tokenizer-fix",
                    "max_model_len": 16_000,
                    "gpu_memory_utilization": 0.6,
                    "eos_token_id": 128009,
                },
                endpoint=cmodels.Endpoints.chat_completions.value,
                checking_function="check_text_result",
                task=CHAT_LLAMA_3_1_70B,
            ),
            synthetic_generation_config=cmodels.SyntheticGenerationConfig(func="generate_chat_synthetic", kwargs={"model": CHAT_LLAMA_3_1_70B}),
            endpoint=cmodels.Endpoints.chat_completions.value,
            volume_to_requests_conversion=300,
            is_stream=True,
            weight=0.2,
            timeout=2,
            enabled=True,
        ),
        CHAT_LLAMA_3_1_8B: cmodels.FullTaskConfig(
            task=CHAT_LLAMA_3_1_8B,
            task_type=cmodels.TaskType.TEXT,
            max_capacity=120_000,
            orchestrator_server_config=cmodels.OrchestratorServerConfig(
                server_needed=cmodels.ServerType.LLM,
                load_model_config={
                    "model": "unsloth/Meta-Llama-3.1-8B-Instruct",
                    "half_precision": True,
                    "tokenizer": "tau-vision/llama-tokenizer-fix",
                    "max_model_len": 20_000,
                    "gpu_memory_utilization": 0.65,
                    "eos_token_id": 128009,
                },
                endpoint=cmodels.Endpoints.chat_completions.value,
                checking_function="check_text_result",
                task=CHAT_LLAMA_3_1_8B,
            ),
            synthetic_generation_config=cmodels.SyntheticGenerationConfig(func="generate_chat_synthetic", kwargs={"model": CHAT_LLAMA_3_1_8B}),
            endpoint=cmodels.Endpoints.chat_completions.value,
            volume_to_requests_conversion=300,
            is_stream=True,
            weight=0.15,
            timeout=2,
            enabled=True,
        ),
        PROTEUS_TEXT_TO_IMAGE: cmodels.FullTaskConfig(
            task=PROTEUS_TEXT_TO_IMAGE,
            task_type=cmodels.TaskType.IMAGE,
            max_capacity=800,
            orchestrator_server_config=cmodels.OrchestratorServerConfig(
                server_needed=cmodels.ServerType.IMAGE,
                load_model_config={},
                checking_function="check_image_result",
                endpoint=cmodels.Endpoints.text_to_image.value,
                task=PROTEUS_TEXT_TO_IMAGE,
            ),
            synthetic_generation_config=cmodels.SyntheticGenerationConfig(
                func="generate_text_to_image_synthetic",
                kwargs={"model": PROTEUS_TEXT_TO_IMAGE},
            ),
            endpoint=cmodels.Endpoints.text_to_image.value,
            volume_to_requests_conversion=10,
            is_stream=False,
            weight=0.1,
            timeout=5,
            enabled=True,
            model_info={"model": "dataautogpt3/ProteusV0.4-Lightning"},
        ),
        PROTEUS_IMAGE_TO_IMAGE: cmodels.FullTaskConfig(
            task=PROTEUS_IMAGE_TO_IMAGE,
            task_type=cmodels.TaskType.IMAGE,
            max_capacity=800,
            orchestrator_server_config=cmodels.OrchestratorServerConfig(
                server_needed=cmodels.ServerType.IMAGE,
                load_model_config={},
                checking_function="check_image_result",
                endpoint=cmodels.Endpoints.image_to_image.value,
                task=PROTEUS_IMAGE_TO_IMAGE,
            ),
            synthetic_generation_config=cmodels.SyntheticGenerationConfig(
                func="generate_image_to_image_synthetic",
                kwargs={"model": PROTEUS_IMAGE_TO_IMAGE},
            ),
            endpoint=cmodels.Endpoints.image_to_image.value,
            volume_to_requests_conversion=10,
            is_stream=False,
            weight=0.05,
            timeout=20,
            enabled=True,
            model_info={"model": "dataautogpt3/ProteusV0.4-Lightning"},
        ),
        FLUX_SCHNELL_TEXT_TO_IMAGE: cmodels.FullTaskConfig(
            task=FLUX_SCHNELL_TEXT_TO_IMAGE,
            task_type=cmodels.TaskType.IMAGE,
            max_capacity=2100,
            orchestrator_server_config=cmodels.OrchestratorServerConfig(
                server_needed=cmodels.ServerType.IMAGE,
                load_model_config={},
                checking_function="check_image_result",
                endpoint=cmodels.Endpoints.text_to_image.value,
                task=FLUX_SCHNELL_TEXT_TO_IMAGE,
            ),
            synthetic_generation_config=cmodels.SyntheticGenerationConfig(
                func="generate_text_to_image_synthetic",
                kwargs={"model": FLUX_SCHNELL_TEXT_TO_IMAGE},
            ),
            endpoint=cmodels.Endpoints.text_to_image.value,
            volume_to_requests_conversion=10,
            is_stream=False,
            weight=0.15,
            timeout=20,
            enabled=True,
            model_info={"model": "black-forest-labs/FLUX.1-schnell"},
        ),
        FLUX_SCHNELL_IMAGE_TO_IMAGE: cmodels.FullTaskConfig(
            task=FLUX_SCHNELL_IMAGE_TO_IMAGE,
            task_type=cmodels.TaskType.IMAGE,
            max_capacity=800,
            orchestrator_server_config=cmodels.OrchestratorServerConfig(
                server_needed=cmodels.ServerType.IMAGE,
                load_model_config={},
                checking_function="check_image_result",
                endpoint=cmodels.Endpoints.image_to_image.value,
                task=FLUX_SCHNELL_IMAGE_TO_IMAGE,
            ),
            synthetic_generation_config=cmodels.SyntheticGenerationConfig(
                func="generate_image_to_image_synthetic",
                kwargs={"model": FLUX_SCHNELL_IMAGE_TO_IMAGE},
            ),
            endpoint=cmodels.Endpoints.image_to_image.value,
            volume_to_requests_conversion=10,
            is_stream=False,
            weight=0.05,
            timeout=15,
            enabled=True,
            model_info={"model": "black-forest-labs/FLUX.1-schnell"},
        ),
        AVATAR: cmodels.FullTaskConfig(
            task=AVATAR,
            task_type=cmodels.TaskType.IMAGE,
            max_capacity=800,
            orchestrator_server_config=cmodels.OrchestratorServerConfig(
                server_needed=cmodels.ServerType.IMAGE,
                load_model_config={},
                checking_function="check_image_result",
                endpoint=cmodels.Endpoints.avatar.value,
                task=AVATAR,
            ),
            synthetic_generation_config=cmodels.SyntheticGenerationConfig(
                func="generate_avatar_synthetic",
                kwargs={},
            ),
            endpoint=cmodels.Endpoints.avatar.value,
            volume_to_requests_conversion=10,
            is_stream=False,
            weight=0.15,
            timeout=15,
            enabled=True,
            model_info={"model": "dataautogpt3/ProteusV0.4-Lightning"},
        ),
        DREAMSHAPER_TEXT_TO_IMAGE: cmodels.FullTaskConfig(
            task=DREAMSHAPER_TEXT_TO_IMAGE,
            task_type=cmodels.TaskType.IMAGE,
            max_capacity=800,
            orchestrator_server_config=cmodels.OrchestratorServerConfig(
                server_needed=cmodels.ServerType.IMAGE,
                load_model_config={},
                checking_function="check_image_result",
                endpoint=cmodels.Endpoints.text_to_image.value,
                task=DREAMSHAPER_TEXT_TO_IMAGE,
            ),
            synthetic_generation_config=cmodels.SyntheticGenerationConfig(
                func="generate_text_to_image_synthetic",
                kwargs={"model": DREAMSHAPER_TEXT_TO_IMAGE},
            ),
            endpoint=cmodels.Endpoints.text_to_image.value,
            volume_to_requests_conversion=10,
            is_stream=False,
            weight=0.05,
            timeout=5,
            enabled=True,
            model_info={"model": "Lykon/dreamshaper-xl-lightning"},
        ),
        DREAMSHAPER_IMAGE_TO_IMAGE: cmodels.FullTaskConfig(
            task=DREAMSHAPER_IMAGE_TO_IMAGE,
            task_type=cmodels.TaskType.IMAGE,
            max_capacity=800,
            orchestrator_server_config=cmodels.OrchestratorServerConfig(
                server_needed=cmodels.ServerType.IMAGE,
                load_model_config={},
                checking_function="check_image_result",
                endpoint=cmodels.Endpoints.image_to_image.value,
                task=DREAMSHAPER_IMAGE_TO_IMAGE,
            ),
            synthetic_generation_config=cmodels.SyntheticGenerationConfig(
                func="generate_image_to_image_synthetic",
                kwargs={"model": DREAMSHAPER_IMAGE_TO_IMAGE},
            ),
            endpoint=cmodels.Endpoints.image_to_image.value,
            volume_to_requests_conversion=10,
            is_stream=False,
            weight=0.05,
            timeout=15,
            enabled=True,
            model_info={"model": "Lykon/dreamshaper-xl-lightning"},
        ),
    }
