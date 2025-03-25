#!/bin/bash
source activate venv_speech
port=${PORT:-6918}

sudo mkdir -p .checkpoints
cd .checkpoints
if [ ! -f imagebind_huge.pth ]; then
    wget https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth
else
    echo ".checkpoints/imagebind_huge.pth already exists. Skipping download."
fi
cd ..
if [ ! -f bpe/bpe_simple_vocab_16e6.txt.gz ]; then
    curl -L -o bpe_simple_vocab_16e6.txt.gz https://github.com/facebookresearch/ImageBind/raw/main/imagebind/bpe/bpe_simple_vocab_16e6.txt.gz
    mkdir -p bpe
    mv bpe_simple_vocab_16e6.txt.gz bpe/
else
    echo "bpe/bpe_simple_vocab_16e6.txt.gz already exists. Skipping download."
fi
sudo rm -r /root/bentoml/models
nohup bentoml serve . -p 5001 > ImageBind.log 2>&1 &

uvicorn main:app --host 0.0.0.0 --port ${port}