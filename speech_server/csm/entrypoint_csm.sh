#!/bin/bash
source activate venv_speech
port=${PORT:-6919}

sudo rm -r /root/bentoml/models

# Huggingface login to download gated CSM and Mimi models
if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN found. Attempting to log in and clone repository..."

    # Check if huggingface-cli is installed
    if ! command_exists huggingface-cli; then
        echo "huggingface-cli is not installed. Installing it now..."
        pip install huggingface_hub
    fi

    # Log in to Huggingface
    echo "Logging in to Huggingface..."
    git config --global credential.helper store
    huggingface-cli login --token $HF_TOKEN --add-to-git-credential

    nohup bentoml serve . -p 5006 > csm.log 2>&1 &
else
    echo "HF_TOKEN not set. Skipping Huggingface login and setup"
fi

uvicorn main:app --host 0.0.0.0 --port ${port}