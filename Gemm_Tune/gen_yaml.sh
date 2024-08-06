#!/usr/bin/bash

set -ex

declare -a MODELS=("NousResearch/Llama-2-7b-chat-hf" "NousResearch/Llama-2-70b-hf" "NousResearch/Meta-Llama-3.1-8B" "NousResearch/Meta-Llama-3.1-70B" "Qwen/Qwen2-7B-Instruct" "Qwen/Qwen2-72B-Instruct" "Qwen/Qwen1.5-110B-Chat")

ROOT_PATH=$(dirname $(dirname "$PWD"))"/"
BENCHMARK_PATH=$ROOT_PATH"Ali_MI300_Tuned/Ali_PoC/throughput/benchmark_throughput_0802_1717.py"
# BENCHMARK_PATH=$ROOT_PATH"/vllm/benchmarks/benchmark_throughput.py"
TEMP_FILE_PATH=$ROOT_PATH"PerfRes/"
KIT_SOURCE_PATH=$ROOT_PATH"Ali_MI300_Tuned/pytorch_afo_testkit"
GEMM_TUNE_PATH=$ROOT_PATH"Ali_MI300_Tuned/Gemm_Tune/"
echo $BENCHMARK_PATH
OUTPUT_LEN=500
NUM_SEQ=1000
INPUT_LEN="1000 2000"

for MODEL in ${MODELS[@]}; do
    if [[ $(echo "${MODEL}" | grep "110B") != "" ]]; then
        CURRTP=2
    elif [[ $(echo "${MODEL}" | grep "70B") != "" ]] || [[ $(echo "${MODEL}" | grep "72B") != "" ]]; then
        CURRTP=2
    else
        CURRTP=1
    fi

    for inp in ${INPUT_LEN}; do
        echo "[LOG] model is ${MODEL}"
        echo "[LOG] tp is ${CURRTP}"
        CURRMODEL=$(echo ${MODEL} | cut -d'/' -f 2)
        echo "[LOG] model is ${CURRMODEL}"
        echo "[LOG] input length is ${inp}"

        KIT_PATH=$ROOT_PATH$CURRMODEL"/"${inp}"/"
        if [ ! -d $KIT_PATH ]; then
            mkdir -p $KIT_PATH
        fi
        if [ ! -d $GEMM_TUNE_PATH$CURRMODEL ]; then 
            mkdir -p $GEMM_TUNE_PATH/$CURRMODEL
        fi
        if [ ! -d $TEMP_FILE_PATH$CURRMODEL ]; then 
            mkdir -p $TEMP_FILE_PATH$CURRMODEL
        fi
        
        if [ "$CURRTP" -eq 1 ]; then
            RUN_SCRIPT="python $BENCHMARK_PATH --backend vllm --input-len ${inp} --output-len $OUTPUT_LEN --num-prompts $NUM_SEQ --model "${MODELS[${i}]}" --tokenizer "${MODELS[${i}]}" --dtype float16 --enforce-eager --quantization-param-path "$TEMP_FILE_PATH" --device cuda --download-dir "$TEMP_FILE_PATH" --output-json "$TEMP_FILE_PATH$CURRMODEL"/input${inp}-output$OUTPUT_LEN-prompt$NUM_SEQ-tp"$CURRTP".json"
        else
            RUN_SCRIPT="torchrun --standalone --nnodes 1 --nproc-per-node $CURRTP $BENCHMARK_PATH --backend vllm --input-len ${inp} --output-len $OUTPUT_LEN --num-prompts $NUM_SEQ --model "${MODELS[${i}]}" --tokenizer "${MODELS[${i}]}" -tp "$CURRTP" --dtype float16 --enforce-eager --quantization-param-path "$TEMP_FILE_PATH" --device cuda --download-dir "$TEMP_FILE_PATH" --output-json "$TEMP_FILE_PATH$CURRMODEL"/input${inp}-output$OUTPUT_LEN-prompt$NUM_SEQ-tp"$CURRTP".json"
        fi

        cd $ROOT_PATH

        cd $KIT_PATH
        YAML_NAME=$CURRMODEL"_"${inp}".yaml"
        echo "[LOG] yaml file: "$YAML_NAME
        ROCBLAS_LAYER=4 $RUN_SCRIPT 2>&1 |  grep "\- { rocblas_function:" | uniq | tee $YAML_NAME        
        
        cp -r $YAML_NAME $GEMM_TUNE_PATH/$CURRMODEL
    done
done