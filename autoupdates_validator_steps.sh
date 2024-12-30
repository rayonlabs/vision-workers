#!/bin/bash

## STEPS TO RUN IN AUTOUPDATES (you can pass on custom docker images - orchestrator, llm, image - )

docker image prune -a -f --filter "until=168h"

./launch_orchestrator.sh