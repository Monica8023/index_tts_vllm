#!/bin/bash
VLLM_USE_V1=0 python3 api_server.py --model_dir assets/checkpoints/ --port 8001