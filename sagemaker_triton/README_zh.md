# Whisper ASR SageMaker Triton 部署

本项目演示了如何使用 NVIDIA Triton 推理服务器在 Amazon SageMaker 上部署 Whisper 自动语音识别（ASR）模型。它提供了一种可扩展且高效的方法，使用 Whisper large-v3 模型进行语音到文本的转录。

## 概述

该项目使用自定义 Docker 容器在 SageMaker 上部署 Whisper 模型，利用 Triton 推理服务器进行优化推理。它包含了模型部署、端点创建和转录测试的脚本。

## 先决条件

- 具有 SageMaker 访问权限的 AWS 账户
- 本地机器上安装的 Docker
- Python 3.7+
- 配置了适当权限的 AWS CLI
- NVIDIA GPU 支持（用于本地测试和开发）

## 设置

1. 克隆此仓库：
   ```
   git clone https://github.com/xqun3/whisper-sagemaker-triton.git
   cd whisper-sagemaker-triton/sagemaker_triton
   ```

2. 构建并将 Docker 镜像推送到 Amazon ECR：
   ```
   ./build_and_push.sh
   ```

3. 合并模型（如果有用到 lora 微调模型，如果没有可跳过此步）并保存成 .pt 格式，执行以下代码，模型路径参数修改为自己的路径

   这里以sagemaker notebook 终端路径为例
   ```
   # 安装相关依赖
   pip install openai-whisper peft transformers

   python merge_lora.py --model-id openai/whisper-large-v3 --lora-path /path/to/lora/checkpoints --export-to /home/ec2-user/SageMaker/whisper-sagemaker-triton/sagemaker_triton/assets/large-v3.pt
   ```

4. 编译模型
   - 进入容器
   ```
   # 将以下 docker 的挂载路径替换成上面模型所在的实际父目录
   docker run --rm -it --net host --shm-size=2g --gpus all \
      -v /home/ec2-user/SageMaker/whisper-sagemaker-triton/sagemaker_triton/assets:/workspace/assets/ \
      -v /home/ec2-user/SageMaker/whisper-sagemaker-triton/sagemaker_triton/:/workspace/ \
      sagemaker-endpoint/whisper-triton-byoc:latest /bin/bash
   ```

   - 编译模型
   ```
   # 进入 docker 环境后编译模型
   bash export_model.sh
   ```

5. 上传编译后的模型到 S3
   ```
   aws s3 sync /home/ec2-user/SageMaker/whisper-sagemaker-triton/sagemaker_triton/model_repo_whisper_trtllm/ s3://<Your S3 path>
   ```

6. 模型部署
   - 修改 model_data/start_triton_and_client.sh 文件，将其里面 s3 下载的模型路径替换成上面实际的 s3 路径
   ```
   python3 download_model_from_s3.py --source_s3_url <s3://<Your S3 path> --local_dir_path /model_repo_whisper_trtllm --working_dir /workspace
   ```
   - 然后在 SageMaker Studio 或 Jupyter 或在已配置好能够访问 aws 服务的本地机器打开 `deploy_and_test_preprocessed.ipynb` notebook
   - 按照 `deploy_and_test_preprocessed.ipynb` notebook 中的代码执行部署

7. 调用测试

   部署后，您可以使用 SageMaker 端点进行转录。notebook 中包含了转录音频文件的示例代码，同时也可以参考 [inference.py](https://github.com/xqun3/whisper-sagemaker-triton/blob/main/sagemaker_triton/inference.py) 进行调用


## Docker 镜像

Docker 镜像基于 NVIDIA Triton 服务器镜像（nvcr.io/nvidia/tritonserver:24.05-py3），并包含以下关键组件：

- 用于音频处理的 FFmpeg
- TensorRT-LLM 0.11.0.dev2024052800
- Triton 客户端库
- 自定义 Python 依赖项（参见 requirements.txt）

## Python 依赖项

该项目所需的关键 Python 包包括：

- sagemaker 和 sagemaker-ssh-helper，用于 SageMaker 集成
- boto3，用于 AWS SDK
- openai-whisper，用于 Whisper ASR 模型
- tritonclient[grpc]，用于 Triton 推理服务器客户端
- fastapi 和 uvicorn，用于 API 服务
- 其他音频处理和机器学习库（librosa、soundfile、transformers 等）

完整列表请参阅 `requirements.txt` 文件。

## 关键组件

- `deploy_and_test_preprocessed.ipynb`：用于部署和测试的主要笔记本
- `Dockerfile.server`：定义用于 SageMaker 部署的自定义容器
- `model_data/`：包含 Triton 服务器设置和客户端交互的脚本
- `requirements.txt`：列出 Python 依赖项
- `build_and_push.sh`：构建并将 Docker 镜像推送到 ECR 的脚本
- `serve`：Docker 容器的入口脚本

## 自定义

您可以通过修改以下内容来自定义部署：
- 笔记本中的 Whisper 模型版本（默认为 whisper-large-v3）
- 部署的实例类型（默认为 ml.g5.xlarge）
- 解码方法和其他推理参数
- `Dockerfile.server` 中的 Docker 镜像配置

## 调试

该项目包括使用 SageMaker SSH Helper 进行 SSH 调试的支持。这允许您连接到 SageMaker 实例进行故障排除和开发。

## 清理

为避免产生不必要的费用，请记得在不再需要时删除 SageMaker 端点、端点配置和模型。您可以使用笔记本中提供的清理代码：

```python
sess.delete_endpoint(endpoint_name)
sess.delete_endpoint_config(endpoint_name)
sess.delete_model(model.name)
```

## 贡献

欢迎为改进项目做出贡献。请遵循标准的 GitHub 拉取请求流程来提出更改。

## 许可证

[指定该项目发布所依据的许可证]

## 致谢

本项目使用了以下开源软件：
- NVIDIA Triton 推理服务器
- OpenAI Whisper
- TensorRT-LLM

请参阅它们各自的许可证以了解使用条款。