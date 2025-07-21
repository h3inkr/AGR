#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python src/generate_expand.py \
    --analyzed_path results/analysis_nq.jsonl \
    --output_path results/generation_nq.jsonl \