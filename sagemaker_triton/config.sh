#!/bin/bash

# 项目配置
# PROJECT_ROOT="/home/ec2-user/SageMaker/whisper-sagemaker-triton"
PROJECT_ROOT="/home/ec2-user/SageMaker/whisper-sagemaker-triton"

#注意 s3 路径最后加上 /
S3_PATH="s3://triton-models-xq/test_1012/"

# Docker 配置
## 注意：如果修了docker image 的值，也需要同步修改 deploy_and_test_preprocessed.ipynb 文件里的镜像地址
## 默认的 image url 会是: ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${DOCKER_IMAGE}
DOCKER_IMAGE="sagemaker-endpoint/whisper-triton-byoc:latest"

# 模型配置
HUGGING_FACE_MODEL_ID="openai/whisper-large-v3"
OUTPUT_MODEL_PATH="$PROJECT_ROOT/sagemaker_triton/assets/large-v3.pt"
OPENAI_WHISPER_DOWNLOAD_URL="https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt"

# LoRA 配置
USE_LORA=false  # 设置为 true 如果需要合并 LoRA 模型
LORA_PATH="/home/ec2-user/SageMaker/test_whisper/lora_model/checkpoint-135"  # 仅在 USE_LORA 为 true 时使用

# 其他配置
CONDA_ENV="pytorch_p310"
