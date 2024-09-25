#!/bin/bash
set -e

git clone https://github.com/NVIDIA/TensorRT-LLM.git -b v0.11.0
cd /workspace/TensorRT-LLM/examples/whisper

# take large-v3 model as an example
# wget --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt

INFERENCE_PRECISION=float16
MAX_BEAM_WIDTH=4
MAX_BATCH_SIZE=48
checkpoint_dir=tllm_checkpoint
output_dir=whisper_large_v3

cp -r /workspace/assets /workspace/TensorRT-LLM/examples/whisper

# Convert the large-v3 openai model into trtllm compatible checkpoint.
python3 convert_checkpoint.py \
                --output_dir $checkpoint_dir

# Build the large-v3 trtllm engines
trtllm-build --checkpoint_dir ${checkpoint_dir}/encoder \
                --output_dir ${output_dir}/encoder \
                --paged_kv_cache disable \
                --moe_plugin disable \
                --enable_xqa disable \
                --use_custom_all_reduce disable \
                --max_batch_size ${MAX_BATCH_SIZE} \
                --gemm_plugin disable \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --remove_input_padding disable

trtllm-build --checkpoint_dir ${checkpoint_dir}/decoder \
                --output_dir ${output_dir}/decoder \
                --paged_kv_cache disable \
                --moe_plugin disable \
                --enable_xqa disable \
                --use_custom_all_reduce disable \
                --max_beam_width ${MAX_BEAM_WIDTH} \
                --max_batch_size ${MAX_BATCH_SIZE} \
                --max_output_len 100 \
                --max_input_len 14 \
                --max_encoder_input_len 1500 \
                --gemm_plugin ${INFERENCE_PRECISION} \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --gpt_attention_plugin ${INFERENCE_PRECISION} \
                --remove_input_padding disable

# prepare the model_repo_whisper_trtllm
cp -r /workspace/TensorRT-LLM/examples/whisper/whisper_large_v3 /workspace/model_repo_whisper_trtllm/whisper/1/
