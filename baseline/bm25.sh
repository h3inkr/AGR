#!/bin/bash
export VLLM_NCCL_SO_PATH=/home/user4/.conda/envs/agr/lib/libnccl.so.2
export JAVA_HOME=$HOME/java/jdk-21.0.1+12
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so
export CONDA_PREFIX=/home/user4/.conda/envs/agr
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/server:$LD_LIBRARY_PATH

DATASET=csqa

python baseline/bm25_retrieval.py \
    --qa_path data/CSQA/test.json \
    --dataset_name $DATASET \
    --passage data/merged_raco.tsv \
    --result_file_path results/$DATASET/bm25_retrieval_$DATASET.json \
    --index_name /home/user4/agr/raco/raco_index \
    --num_threads 32 \
    --dedup

python src/inference.py \
    --eval_path results/$DATASET/bm25_retrieval_$DATASET.json

# Natural Questions
#🔑 Hit@5: 43.77%
#🔑 Hit@100: 78.23%

# 2WikiMultiHop
#🔑 Hit@5: 30.29%
#🔑 Hit@100: 57.97%

# CSQA (Wikipedia corpus)
#🔑 Hit@5: 25.55%
#🔑 Hit@100: 79.28%

# CSQA (RaCO)
#🔑 Hit@5: 17.2%
#🔑 Hit@100: 21.21%