#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python src/generate_enriched.py \
    --retrieved_path results/nq/retrived_results_nq.json \
    --output_path results/nq/enriched_nq.jsonl \