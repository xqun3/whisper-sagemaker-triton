#!/bin/bash
#set -e
echo "Current working directory: $(pwd)"

# Check if TensorRT-LLM directory exists and remove it if it does
if [ -d "TensorRT-LLM" ]; then
    echo "TensorRT-LLM directory already exists. Removing it..."
    rm -rf TensorRT-LLM
fi

git clone https://github.com/NVIDIA/TensorRT-LLM.git 
cd /workspace/TensorRT-LLM/examples/whisper


INFERENCE_PRECISION=float16
MAX_BEAM_WIDTH=4
MAX_BATCH_SIZE=48
checkpoint_dir=tllm_checkpoint
output_dir=whisper_large_v3_turbo
model_name=$1
echo "模型名称: $model_name"

cp -r /workspace/assets /workspace/TensorRT-LLM/examples/whisper

# Convert the large-v3 openai model into trtllm compatible checkpoint.
python3 convert_checkpoint.py \
                --model_name $model_name \
                --output_dir $checkpoint_dir

# Build the large-v3 trtllm engines
trtllm-build --checkpoint_dir ${checkpoint_dir}/encoder \
                --output_dir ${output_dir}/encoder \
                --paged_kv_cache enable \
                --moe_plugin disable \
                --enable_xqa disable \
                --max_batch_size ${MAX_BATCH_SIZE} \
                --gemm_plugin disable \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --remove_input_padding enable \
		--max_input_len 3000 --max_seq_len 3000

trtllm-build --checkpoint_dir ${checkpoint_dir}/decoder \
                --output_dir ${output_dir}/decoder \
                --paged_kv_cache enable \
                --moe_plugin disable \
                --enable_xqa disable \
                --max_beam_width ${MAX_BEAM_WIDTH} \
                --max_batch_size ${MAX_BATCH_SIZE} \
                --max_seq_len 114 \
                --max_input_len 14 \
                --max_encoder_input_len 3000\
                --gemm_plugin ${INFERENCE_PRECISION} \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --gpt_attention_plugin ${INFERENCE_PRECISION} \
                --remove_input_padding enable 

ls ${output_dir}
# prepare the model_repo_whisper_trtllm
model_repo=/workspace/whisper_trt/
rm -rf $model_repo 
mkdir $model_repo  
cp -r /workspace/model_repo_whisper_trtllm_ifb/* $model_repo
echo "fill template"
wget --directory-prefix=$model_repo/infer_bls/1 https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
wget --directory-prefix=$model_repo/whisper/1 assets/mel_filters.npz https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz

rm -rf /workspace/whisper_turbo
mv ${output_dir} /workspace/whisper_turbo
output_dir=/workspace/whisper_turbo
ls ${output_dir}
n_mels=128
zero_pad=false

TRITON_MAX_BATCH_SIZE=48
MAX_QUEUE_DELAY_MICROSECONDS=100
python3 /workspace/fill_template.py -i $model_repo/whisper/config.pbtxt engine_dir:${output_dir},n_mels:$n_mels,zero_pad:$zero_pad,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
cat $model_repo/whisper/config.pbtxt 

echo "fill infer_bls"
python3 /workspace/fill_template.py -i $model_repo/infer_bls/config.pbtxt engine_dir:${output_dir},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
cat $model_repo/infer_bls/config.pbtxt

#cp -r /workspace/TensorRT-LLM/examples/whisper/whisper_large_v3_turbo /workspace/model_repo_whisper_trtllm/whisper/1/

rm -rf /workspace/TensorRT-LLM
echo "export model done"

