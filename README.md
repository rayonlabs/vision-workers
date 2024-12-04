<div align="center">

# **👀 Vision Workers [τ, τ] SN19: Starter Guide for Miners & Validators**
Providing Access to Bittensor with Decentralized Subnet Inference at Scale

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Bittensor](https://bittensor.com/whitepaper) • [Discord](https://discord.gg/dR865yTPaZ) • [Corcel](https://app.corcel.io/studio) • [Nineteen](https://nineteen.ai/)
</div>

## NOTE:
PLEASE MAKE ANY PR'S TO THE *PRE-MAIN* BRANCH FIRST.

This way we can build the images before merging to main

## Introduction

Welcome to sn19 GPU stuff! Follow the below guides to get started

PS : After this, you still need to run proxy script using instructions in [Proxy setup](https://github.com/namoray/vision)

# Validators 👨‍⚖️

Validators should use bare metal machines due to the use of Docker-in-Docker for job orchestration. Cloud providers like Runpod or VastAI will not work for this setup.

1. [Recommened compute](recommend-compute.md)
2. [Docs to get started](validator_orchestrator/docs/README.md)

# Miners ⛏️

Miner workers can operate on both Runpod/VastAI and bare metal machines, although bare metal is recommended for better performance and stability.

## For Runpod and Similar Providers

Refer to the [Runpod Setup Guide](https://github.com/namoray/vision/blob/production/docs/mining.md#worker-server-setup) for instructions on starting workers.

## For Bare Metal Users

Refer to the following guides:
1. [Setting Up Your Hardware](generic_docs/bootstrap.md)
2. [Image Server Installation](image_server/docs/installation_base.md)
3. [LLM Server Installation](llm_server/docs/installation_base.md)

# Conclusion

Please refer to the [Proxy setup](https://github.com/namoray/vision/tree/main/docs) page in order to set up the proxy.

For further assistance, refer to the [Bittensor Documentation](https://docs.bittensor.com/) and join our [Discord](https://discord.gg/dR865yTPaZ).

Happy Mining and Validating!
