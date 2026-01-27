#!/bin/bash

MODEL_NAME=configs/standard/qwen3_8b.yaml

python3 agent.py \
    --yaml_name $MODEL_NAME

