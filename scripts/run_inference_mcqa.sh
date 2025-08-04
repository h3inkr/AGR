#!/bin/bash
export VLLM_NCCL_SO_PATH=/home/user4/.conda/envs/agr/lib/libnccl.so.2
export JAVA_HOME=$HOME/java/jdk-21.0.1+12
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so
export CONDA_PREFIX=/home/user4/.conda/envs/agr
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/server:$LD_LIBRARY_PATH
export DATASET=arc

# Retrieval for evaluation
CUDA_VISIBLE_DEVICES=3 python src/evaluate_retrieval_mcqa.py \
    --qa_path results/${DATASET}/refined_${DATASET}.jsonl \
    --data_path data/ARC-C/test.json \
    --dataset_name $DATASET \
    --passage data/psgs_w100.tsv \
    --result_file_path results/${DATASET}/eval_retrieval_${DATASET}.json \
    --index_name "wikipedia-dpr" \
    --num_threads 32 \
    --dedup
