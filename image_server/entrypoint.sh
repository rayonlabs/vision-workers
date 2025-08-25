#!/bin/bash
cleanup() {
    echo "Stopping the ComfyUI server..."
    kill $COMFY_SERVER_PID
    wait $COMFY_SERVER_PID 2>/dev/null
    echo "Both servers have been stopped."
}

trap cleanup SIGINT SIGTERM

echo "Starting the image server. First i'll download all the models..."

python setup.py

vram_mode=${VRAM_MODE:-'--lowvram'}
warmup=$(echo ${WARMUP:-false} | tr '[:upper:]' '[:lower:]')
device=${DEVICE:-0}
port=${PORT:-6919}

cd ComfyUI
if [ -n "$vram_mode" ]
then
    python main.py $vram_mode --cuda-device $device --disable-xformers &
else
    python main.py --disable-xformers  --cuda-device $device &
fi
cd ..

COMFY_SERVER_PID=$!
echo "ComfyUI server started with PID: $COMFY_SERVER_PID"
sleep 5

if [ "$warmup" = "true" ]
then
    python warmup.py
else
    sleep 1
fi

uvicorn main:app --host 0.0.0.0 --port $port
cleanup