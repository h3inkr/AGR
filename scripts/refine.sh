#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python src/refine.py \
    --candidate_path results/nq/enriched_nq.jsonl \
    --output_path results/nq/refined_nq.jsonl \