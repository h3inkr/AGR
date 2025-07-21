#!/bin/bash
export DATASET=nq

python src/inference.py \
    --eval_path results/${DATASET}/eval_retrieval_${DATASET}.json \