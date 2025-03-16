
# Vision Orchestrator Server Setup

To run the server on bare metal, you will need Nvidia drivers, CUDA toolkit, and Docker.

## Prerequisites

1. [Prerequisites](../../../generic_docs/prerequisites.md)

## Running the Server

The server uses port 6919 to communicate with the LLM and image servers it spins up locally.

**Note:** Ensure that port 6919 is not used by another application on this GPU!

Port 6919 does not need to be exposed. Instead, port 6920 should be exposed (or any other port you choose if you run using a different one).


## To manually start 
If you used the bootstrap script you should be good to go already (check with docker ps). 

If you need to start it manually at any time, try this:
```bash
python3 create_config.py #to create config
./launch_orchestrator.sh
```



For further troubleshooting, refer to the [troubleshooting guide](../../../generic_docs/troubleshooting.md).
