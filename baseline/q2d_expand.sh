#!/bin/bash
DATASET=csqa

CUDA_VISIBLE_DEVICES=1 python baseline/q2d_expand.py \
    --data_path data/CSQA/test.json\
    --output_path results/$DATASET/q2d_expanded.jsonl\
    --n 1