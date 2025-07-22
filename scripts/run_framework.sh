#!/bin/bash
export DATASET=arc
export VLLM_NCCL_SO_PATH=/home/user4/.conda/envs/agr/lib/libnccl.so.2
export JAVA_HOME=$HOME/java/jdk-21.0.1+12
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so
export CONDA_PREFIX=/home/user4/.conda/envs/agr
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/server:$LD_LIBRARY_PATH

# Analyze
CUDA_VISIBLE_DEVICES=1 python src/analyze.py \
    --data_path data/ARC-C/test.json\
    --output_path results/${DATASET}/analysis_${DATASET}.jsonl\

# Generate - Expansion
CUDA_VISIBLE_DEVICES=1 python src/generate_expand.py \
    --analyzed_path results/${DATASET}/analysis_${DATASET}.jsonl \
    --output_path results/${DATASET}/generation_${DATASET}.jsonl \

# Generate - Retrieval
CUDA_VISIBLE_DEVICES=1 python src/generate_retrieval.py \
    --expanded_path results/${DATASET}/generation_${DATASET}.jsonl \
    --passage data/psgs_w100.tsv \
    --result_file_path results/${DATASET}/retrived_results_${DATASET}.json \
    --index_name "wikipedia-dpr" \
    --num_threads 32 \
    --dedup

# Generation - Enriched Expansion
CUDA_VISIBLE_DEVICES=1 python src/generate_enriched.py \
    --retrieved_path results/${DATASET}/retrived_results_${DATASET}.json \
    --output_path results/${DATASET}/enriched_${DATASET}.jsonl \

# Refine
CUDA_VISIBLE_DEVICES=1 python src/refine.py \
    --candidate_path results/${DATASET}/enriched_${DATASET}.jsonl \
    --output_path results/${DATASET}/refined_${DATASET}.jsonl \