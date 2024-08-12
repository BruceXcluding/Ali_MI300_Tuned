#!/usr/bin/bash

set -ex

# declare -a MODELS=("NousResearch/Llama-2-7b-chat-hf" "NousResearch/Llama-2-70b-hf" "NousResearch/Meta-Llama-3.1-8B" "NousResearch/Meta-Llama-3.1-70B" "Qwen/Qwen2-7B-Instruct" "Qwen/Qwen2-72B-Instruct" "Qwen/Qwen1.5-110B-Chat")
declare -a MODELS=("meta-llama/Meta-Llama-3-8B" "meta-llama/Meta-Llama-3-70B" "NousResearch/Llama-2-70b-hf" "NousResearch/Llama-2-7b-chat-hf" "Qwen/Qwen2-7B-Instruct" "Qwen/Qwen2-72B-Instruct" "Qwen/Qwen1.5-110B-Chat")

ROOT_PATH=$(dirname $(dirname "$PWD"))"/"
BENCHMARK_PATH=$ROOT_PATH"Ali_MI300_Tuned/Ali_PoC/throughput/benchmark_throughput_cust_0613_043.py"
# BENCHMARK_PATH=$ROOT_PATH"/vllm/benchmarks/benchmark_throughput.py"
TEMP_FILE_PATH=$ROOT_PATH"PerfRes/"
KIT_SOURCE_PATH=$ROOT_PATH"Ali_MI300_Tuned/pytorch_afo_testkit"
GEMM_TUNE_PATH=$ROOT_PATH"Ali_MI300_Tuned/Gemm_Tune/"
echo $BENCHMARK_PATH
OUTPUT_LEN=500
NUM_SEQ=1000
INPUT_LEN="1000 2000"
export HF_HUB_CACHE="/workspace/PerfRes"

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
        if [ ! -d $GEMM_TUNE_PATH/$CURRMODEL"-"$inp"inp" ]; then 
            mkdir -p $GEMM_TUNE_PATH/$CURRMODEL-${inp}inp
        fi
        if [ ! -d $TEMP_FILE_PATH$CURRMODEL ]; then 
            mkdir -p $TEMP_FILE_PATH$CURRMODEL
        fi
        
        # define json name
        if [[ $1 == "enablehp_tune" ]]; then
            JSON_NAME=hpblas_tuned_input${inp}-output$OUTPUT_LEN-prompt$NUM_SEQ-tp$CURRTP.json
        elif [[ $1 == "enablehp_notune" ]]; then
            JSON_NAME=hpblas_enable_input${inp}-output$OUTPUT_LEN-prompt$NUM_SEQ-tp$CURRTP.json
        elif [[ $1 == "disablehp_tune" ]]; then
            JSON_NAME=dishpblas_tuned_input${inp}-output$OUTPUT_LEN-prompt$NUM_SEQ-tp$CURRTP.json
        else
            JSON_NAME=input${inp}-output$OUTPUT_LEN-prompt$NUM_SEQ-tp$CURRTP.json
        fi
        echo "[LOG] json name is ${JSON_NAME}"

        if [ $CURRTP -eq 1 ]; then
            RUN_SCRIPT="python $BENCHMARK_PATH --backend vllm --input-len ${inp} --output-len $OUTPUT_LEN --num-prompts $NUM_SEQ --model "${MODELS[${i}]}" --tokenizer "${MODELS[${i}]}" --dtype float16 --device cuda --output-json "$TEMP_FILE_PATH$CURRMODEL"/"$JSON_NAME
        else
            RUN_SCRIPT="torchrun --standalone --nnodes 1 --nproc-per-node $CURRTP $BENCHMARK_PATH --backend vllm --input-len ${inp} --output-len $OUTPUT_LEN --num-prompts $NUM_SEQ --model "${MODELS[${i}]}" --tokenizer "${MODELS[${i}]}" -tp "$CURRTP" --dtype float16  --device cuda --output-json "$TEMP_FILE_PATH$CURRMODEL"/"$JSON_NAME
        fi

        cd $ROOT_PATH

        if [[ $1 == "enablehp_tune" ]]; then
            # generate tuned json
            cd $GEMM_TUNE_PATH/$CURRMODEL-${inp}inp
            export PYTORCH_TUNABLEOP_FILENAME=full_tuned%d.csv
            export PYTORCH_TUNABLEOP_NUMERICAL_CHECK=0
            export TORCH_BLAS_PREFER_HIPBLASLT=1
            export PYTORCH_TUNABLEOP_TUNING=0
            export PYTORCH_TUNABLEOP_ENABLED=1
            $RUN_SCRIPT 
            
            #compare json
            cd $TEMP_FILE_PATH$CURRMODEL
            TUNED_THROUGHPUT=$(cat hpblas_tuned_input${inp}*.json | grep tokens_per_second| cut -d":" -f 2| cut -d"," -f 1)
            BEFORETUNED_THROUGHPUT=$(cat hpblas_enable_input${inp}*.json | grep tokens_per_second| cut -d":" -f 2| cut -d"," -f 1)
            echo "Before tuned enablehp: $BEFORETUNED_THROUGHPUT" 
            echo "After tuned enablehp: $TUNED_THROUGHPUT" 
            if [[ $TUNED_THROUGHPUT > $BEFORETUNED_THROUGHPUT ]]; then
                echo "$MODEL inp$inp enablehp: fine tuned"
            else
                echo "$MODEL inp$inp enablehp: bad tuned"
            fi
        elif [[ $1 == "enablehp_notune" ]]; then
            # generate tuned json
            cd $GEMM_TUNE_PATH/$CURRMODEL-${inp}inp
            unset PYTORCH_TUNABLEOP_FILENAME
            export PYTORCH_TUNABLEOP_NUMERICAL_CHECK=0
            export TORCH_BLAS_PREFER_HIPBLASLT=1
            unset PYTORCH_TUNABLEOP_TUNING
            export PYTORCH_TUNABLEOP_ENABLED=0
            $RUN_SCRIPT 
        elif [[ $1 == "disablehp_tune" ]]; then
            # generate tuned json
            cd $GEMM_TUNE_PATH/$CURRMODEL-${inp}inp
            export PYTORCH_TUNABLEOP_FILENAME=full_tuned%d.csv
            export PYTORCH_TUNABLEOP_NUMERICAL_CHECK=0
            export TORCH_BLAS_PREFER_HIPBLASLT=0
            export PYTORCH_TUNABLEOP_TUNING=0
            export PYTORCH_TUNABLEOP_ENABLED=1
            $RUN_SCRIPT 
            
            #compare json
            cd $TEMP_FILE_PATH$CURRMODEL
            TUNED_THROUGHPUT=$(cat hpblas_tuned_input${inp}*.json | grep tokens_per_second| cut -d":" -f 2| cut -d"," -f 1)
            BEFORETUNED_THROUGHPUT=$(cat hpblas_enable_input${inp}*.json | grep tokens_per_second| cut -d":" -f 2| cut -d"," -f 1)
            echo "Before tuned disablehp: $BEFORETUNED_THROUGHPUT" 
            echo "After tuned disablehp: $TUNED_THROUGHPUT" 
            if [[ $TUNED_THROUGHPUT > $BEFORETUNED_THROUGHPUT ]]; then
                echo "$MODEL inp$inp disablehp: fine tuned"
            else
                echo "$MODEL inp$inp disablehp: bad tuned"
            fi
        else
            # generate yaml file
            cd $KIT_PATH
            YAML_NAME=$CURRMODEL"_"${inp}".yaml"
            echo "[LOG] yaml file: "$YAML_NAME

            if [[ $1 == "stop_yaml" ]]; then
                $RUN_SCRIPT
            else
                ROCBLAS_LAYER=4 $RUN_SCRIPT 2>&1 |  grep "\- { rocblas_function:" | uniq | tee $YAML_NAME                    
                cp -r $YAML_NAME $GEMM_TUNE_PATH/$CURRMODEL-${inp}inp
            fi
        fi
        cd $ROOT_PATH
        rm -rf $ROOT_PATH$CURRMODEL/
    done
done
