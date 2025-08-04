#!/bin/bash
export VLLM_NCCL_SO_PATH=/home/user4/.conda/envs/agr/lib/libnccl.so.2
export JAVA_HOME=$HOME/java/jdk-21.0.1+12
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so
export CONDA_PREFIX=/home/user4/.conda/envs/agr
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/server:$LD_LIBRARY_PATH

DATASET="csqa"

CUDA_VISIBLE_DEVICES=3 python baseline/q2d_retrieval.py \
    --qa_path results/$DATASET/q2d_expanded.jsonl \
    --dataset_name $DATASET \
    --passage data/merged_raco.tsv \
    --result_file_path results/$DATASET/q2d_retrieval.json \
    --index_name /home/user4/agr/raco/raco_index \
    --num_threads 32 \
    --dedup

python src/inference.py \
    --eval_path results/$DATASET/q2d_retrieval.json

# NaturalQuestions
#🔑 Hit@5: 61.22%
#🔑 Hit@100: 84.63%

# 2WikiMultiHopQA
#🔑 Hit@5: 30.29%
#🔑 Hit@100: 57.97%