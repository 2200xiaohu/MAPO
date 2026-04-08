#!/bin/bash 

# conda activate naifan_agent

python -m sglang.launch_server \
  --model-path /nas/models/Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --tp-size=8 \
  --port 22222 \
  --trust-remote-code \
  --skip-server-warmup
