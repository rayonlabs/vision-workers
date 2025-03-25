- [ X ] add HF API key functionality for accessing gated models (CSM and Mimi)
- [ X ] verify CSM prototype container with recommended CUDA 12.4 version
- [ X ] verify seed functionality in CSM server 

- [] integrate CSM with nineteen code 
- [] add retry mechanism between container & BentoML interface (/healthz endpoint is deprecated though) 

- [] verify checking reliability with ImageBind 
    - [] Add Imagebind server once CSM prototype done 

- [] add synthetic data generation in nineteen code 
- [] implement speech checking in orchestrator using Imagebind container 
- [] implement 400 HTTP status erroring when tts reference speech input is too long in nineteen code 
