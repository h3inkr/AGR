#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python src/analyze.py \
    --data_path data/NaturalQuestions/test.json\
    --output_path results/analysis_nq.jsonl\