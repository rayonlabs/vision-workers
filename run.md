```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 6919:6919 \
    --ipc=host \
    vllm/vllm-openai:v0.6.3 \
    --model unsloth/Llama-3.2-3b-Instruct \
    --max_model_len 4096 \
    --gpu_memory_utilization 0.7
```

