#!/bin/bash
export DATASET=nq
export VLLM_NCCL_SO_PATH=/home/user4/.conda/envs/agr/lib/libnccl.so.2
export JAVA_HOME=$HOME/java/jdk-21.0.1+12
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so
export CONDA_PREFIX=/home/user4/.conda/envs/agr
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/server:$LD_LIBRARY_PATH

# Generation - Enriched Expansion
CUDA_VISIBLE_DEVICES=3 python src/generate_enriched.py \
    --retrieved_path results/${DATASET}/retrived_results_${DATASET}.json \
    --output_path results/${DATASET}/enriched_${DATASET}.jsonl \

# Refine
CUDA_VISIBLE_DEVICES=3 python src/refine.py \
    --candidate_path results/${DATASET}/enriched_${DATASET}.jsonl \
    --output_path results/${DATASET}/refined_${DATASET}.jsonl \

# Retrieval for evaluation
CUDA_VISIBLE_DEVICES=3 python src/evaluate_retrieval.py \
    --qa_path results/${DATASET}/refined_${DATASET}.jsonl \
    --dataset_name $DATASET \
    --passage data/psgs_w100.tsv \
    --result_file_path results/${DATASET}/eval_retrieval_${DATASET}.json \
    --index_name "wikipedia-dpr" \
    --num_threads 32 \
    --dedup

# Measure Hit@k and EM@k
python src/inference.py \
    --eval_path results/${DATASET}/eval_retrieval_${DATASET}.json 

#🔑 Hit@5: 61.67%
#🔑 Hit@100: 87.22%