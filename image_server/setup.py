"""FOR SOME REASON, DOESN'T WORK GREAT INSIDE DOCKER, DONT USE."""

import os
import subprocess
import shutil
from typing import Optional
from huggingface_hub import hf_hub_download
from loguru import logger


def clone_repo(repo_url: str, target_dir: str, commit_hash: Optional[str] = None) -> None:
    if not os.path.exists(target_dir) or not os.listdir(target_dir):
        subprocess.run(["git", "clone", "--depth", "1", repo_url, target_dir], check=True)
        if commit_hash:
            subprocess.run(["git", "fetch", "--depth", "1", "origin", commit_hash], cwd=target_dir, check=True)
            subprocess.run(["git", "checkout", commit_hash], cwd=target_dir, check=True)


def download_file(*, repo_id: str, filename: str, local_dir: str, cache_dir: str) -> None:
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        cache_dir=cache_dir,
    )

    if os.path.sep in filename:
        flat_target_path = os.path.join(local_dir, os.path.basename(filename))
        shutil.move(downloaded_path, flat_target_path)
        subdir = os.path.join(local_dir, os.path.dirname(filename))
        try:
            while subdir != local_dir:
                os.rmdir(subdir)
                subdir = os.path.dirname(subdir)
        except OSError:
            pass 

        print(f"Moved to: {flat_target_path}")
    else:
        print(f"Downloaded to: {downloaded_path}")


def main():
    logger.info("Cloning ComfyUI...")
    clone_repo(
        repo_url="https://github.com/comfyanonymous/ComfyUI.git",
        target_dir="ComfyUI",
        commit_hash="d9277301d28e732e82d0de1d5948aa00acbf6b65",
    )

    logger.info("Cloned ComfyUI, now downloading all models... (this might take a while)")

    logger.info("Downloading juggerinpaint.safetensors")
    download_file(
        repo_id="tau-vision/jugger-inpaint",
        filename="juggerinpaint.safetensors",
        local_dir="ComfyUI/models/checkpoints",
        cache_dir="ComfyUI/models/caches",
    )

    logger.info("Downloading DreamShaperXL_Turbo_v2_1.safetensors")
    download_file(
        repo_id="Lykon/dreamshaper-xl-v2-turbo",
        filename="DreamShaperXL_Turbo_v2_1.safetensors",
        local_dir="ComfyUI/models/checkpoints",
        cache_dir="ComfyUI/models/caches",
    )

    logger.info("Downloading ProteusV0.4-Lighting.safetensors")
    download_file(
        repo_id="dataautogpt3/ProteusV0.4-Lightning",
        filename="ProteusV0.4-Lighting.safetensors",
        local_dir="ComfyUI/models/checkpoints",
        cache_dir="ComfyUI/models/caches",
    )

    logger.info("Downloading playground-v2.5-1024px-aesthetic.fp16.safetensors")
    download_file(
        repo_id="playgroundai/playground-v2.5-1024px-aesthetic",
        filename="playground-v2.5-1024px-aesthetic.fp16.safetensors",
        local_dir="ComfyUI/models/checkpoints",
        cache_dir="ComfyUI/models/caches",
    )

    logger.info("Downloading t5xxl_fp16.safetensors")
    download_file(
        repo_id="comfyanonymous/flux_text_encoders",
        filename="t5xxl_fp16.safetensors",
        local_dir="ComfyUI/models/text_encoders",
        cache_dir="ComfyUI/models/caches",
    )

    logger.info("Downloading t5xxl_fp8_e4m3fn.safetensors")
    download_file(
        repo_id="comfyanonymous/flux_text_encoders",
        filename="t5xxl_fp8_e4m3fn.safetensors",
        local_dir="ComfyUI/models/text_encoders",
        cache_dir="ComfyUI/models/caches",
    )

    logger.info("Downloading clip_l.safetensors")
    download_file(
        repo_id="comfyanonymous/flux_text_encoders",
        filename="clip_l.safetensors",
        local_dir="ComfyUI/models/text_encoders",
        cache_dir="ComfyUI/models/caches",
    )

    logger.info("Downloading ae.safetensors")
    download_file(
        repo_id="tripathiarpan20/FLUX.1-schnell",
        filename="ae.safetensors",
        local_dir="ComfyUI/models/vae",
        cache_dir="ComfyUI/models/caches",
    )

    logger.info("Downloading flux1-schnell.safetensors")
    download_file(
        repo_id="tripathiarpan20/FLUX.1-schnell",
        filename="flux1-schnell.safetensors",
        local_dir="ComfyUI/models/unet",
        cache_dir="ComfyUI/models/caches",
    )

    logger.info("flux1-kontext-dev.safetensors")
    download_file(
        repo_id="diagonalge/kontext",
        filename="flux1-kontext-dev.safetensors",
        local_dir="ComfyUI/models/diffusion_models",
        cache_dir="ComfyUI/models/caches",
    )

    logger.info("Downloading negativeXL_A.safetensors")
    download_file(
        repo_id="gsdf/CounterfeitXL",
        filename="embeddings/negativeXL_A.safetensors",
        local_dir="ComfyUI/models/embeddings",
        cache_dir="ComfyUI/models/caches",
    )

    logger.info("Downloading sdxl.vae.safetensors")
    download_file(
        repo_id="madebyollin/sdxl-vae-fp16-fix",
        filename="sdxl.vae.safetensors",
        local_dir="ComfyUI/models/vae",
        cache_dir="ComfyUI/models/caches",
    )

    logger.info("Downloaded all models! now setting up ComfyUI...")

    if not os.path.exists("ComfyUI/input/init.png"):
        os.rename("ComfyUI/input/example.png", "ComfyUI/input/init.png")
    if not os.path.exists("ComfyUI/input/mask.png"):
        subprocess.run(["cp", "ComfyUI/input/init.png", "ComfyUI/input/mask.png"], check=True)

    logger.info("Cloning ComfyUI-Inspire-Pack")
    clone_repo(
        repo_url="https://github.com/ltdrdata/ComfyUI-Inspire-Pack",
        target_dir="ComfyUI/custom_nodes/ComfyUI-Inspire-Pack",
        commit_hash="985f6a239b1aed0c67158f64bf579875ec292cb2",
    )

    logger.info("Cloning ComfyUI_IPAdapter_plus")
    clone_repo(
        repo_url="https://github.com/cubiq/ComfyUI_IPAdapter_plus",
        target_dir="ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus",
        commit_hash="0d0a7b3693baf8903fe2028ff218b557d619a93d",
    )

    logger.info("Cloning ComfyUI_InstantID")
    clone_repo(
        repo_url="https://github.com/cubiq/ComfyUI_InstantID",
        target_dir="ComfyUI/custom_nodes/ComfyUI_InstantID",
        commit_hash="50445991e2bd1d5ec73a8633726fe0b33a825b5b",
    )

    logger.info("Downloading antelopev2.zip")
    download_file(
        repo_id="tau-vision/insightface-antelopev2",
        filename="antelopev2.zip",
        local_dir="ComfyUI/models/insightface/models",
        cache_dir="ComfyUI/models/caches",
    )
    if not os.path.exists("ComfyUI/models/insightface/models/antelopev2"):
        subprocess.run(
            ["unzip", "ComfyUI/models/insightface/models/antelopev2.zip", "-d", "ComfyUI/models/insightface/models"],
            check=True,
        )

    logger.info("Downloading ip-adapter.bin")
    download_file(
        repo_id="InstantX/InstantID",
        filename="ip-adapter.bin",
        local_dir="ComfyUI/models/instantid",
        cache_dir="ComfyUI/models/caches",
    )

    logger.info("Downloading diffusion_pytorch_model.safetensors")
    download_file(
        repo_id="InstantX/InstantID",
        filename="ControlNetModel/diffusion_pytorch_model.safetensors",
        local_dir="ComfyUI/models/controlnet",
        cache_dir="ComfyUI/models/caches",
    )

    print("Setup completed successfully.")


if __name__ == "__main__":
    main()
