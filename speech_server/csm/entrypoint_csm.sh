#!/bin/bash
source activate venv_speech
port=${PORT:-6919}

sudo rm -r /root/bentoml/models
nohup bentoml serve . -p 5006 > csm.log 2>&1 &

uvicorn main:app --host 0.0.0.0 --port ${port}