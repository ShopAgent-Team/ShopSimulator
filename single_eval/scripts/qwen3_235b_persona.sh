#!/bin/bash

MODEL_NAME=configs/persona/qwen3_235b.yaml

python3 agent.py \
    --yaml_name $MODEL_NAME
