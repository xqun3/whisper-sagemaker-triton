# Whisper ASR SageMaker Triton Deployment

This project demonstrates the deployment of the Whisper Automatic Speech Recognition (ASR) model on Amazon SageMaker using NVIDIA Triton Inference Server. It provides a scalable and efficient way to perform speech-to-text transcription using the Whisper large-v3 model.

## Overview

The project uses a custom Docker container to deploy the Whisper model on SageMaker, leveraging the Triton Inference Server for optimized inference. It includes scripts for model deployment, endpoint creation, and transcription testing.

## Prerequisites

- An AWS account with SageMaker access
- Docker installed on your local machine
- Python 3.7+
- AWS CLI configured with appropriate permissions
- NVIDIA GPU support (for local testing and development)

## Setup

1. Clone this repository:
   ```
   git clone <repository-url>
   cd whisper-sagemaker-triton/sagemaker_triton
   ```

2. Build and push the Docker image to Amazon ECR:
   ```
   ./build_and_push.sh
   ```

## Docker Image

The Docker image is based on the NVIDIA Triton Server image (nvcr.io/nvidia/tritonserver:24.05-py3) and includes the following key components:

- FFmpeg for audio processing
- TensorRT-LLM 0.11.0.dev2024052800
- Triton Client libraries
- Custom Python dependencies (see requirements.txt)

## Python Dependencies

Key Python packages required for this project include:

- sagemaker and sagemaker-ssh-helper for SageMaker integration
- boto3 for AWS SDK
- openai-whisper for the Whisper ASR model
- tritonclient[grpc] for Triton Inference Server client
- fastapi and uvicorn for API serving
- Other audio processing and machine learning libraries (librosa, soundfile, transformers, etc.)

For a complete list, refer to the `requirements.txt` file.

## Deployment

1. Open the `deploy_and_test_preprocessed.ipynb` notebook in SageMaker Studio or Jupyter.

2. Follow the notebook steps to:
   - Configure SageMaker session and roles
   - Upload the model artifacts to S3
   - Deploy the model to a SageMaker endpoint

## Usage

After deployment, you can use the SageMaker endpoint for transcription. The notebook includes example code for transcribing audio files:

```python
audio_path = "./English_04.wav"
endpoint_name = "<your-endpoint-name>"
result = transcribe_audio(audio_path, endpoint_name)
print(result)
```

## Key Components

- `deploy_and_test_preprocessed.ipynb`: Main notebook for deployment and testing
- `Dockerfile.server`: Defines the custom container for SageMaker deployment
- `model_data/`: Contains scripts for Triton server setup and client interactions
- `requirements.txt`: Lists Python dependencies
- `build_and_push.sh`: Script to build and push the Docker image to ECR
- `serve`: Entry point script for the Docker container

## Customization

You can customize the deployment by modifying:
- The Whisper model version in the notebook (default is whisper-large-v3)
- Instance type for deployment (default is ml.g5.xlarge)
- Decoding method and other inference parameters
- Docker image configuration in `Dockerfile.server`

## Debugging

The project includes support for SSH debugging using the SageMaker SSH Helper. This allows you to connect to the SageMaker instance for troubleshooting and development purposes.

## Cleanup

To avoid incurring unnecessary charges, remember to delete the SageMaker endpoint, endpoint configuration, and model when they are no longer needed. You can use the provided cleanup code in the notebook:

```python
sess.delete_endpoint(endpoint_name)
sess.delete_endpoint_config(endpoint_name)
sess.delete_model(model.name)
```

## Contributing

Contributions to improve the project are welcome. Please follow the standard GitHub pull request process to propose changes.

## License

[Specify the license under which this project is released]

## Acknowledgements

This project uses the following open-source software:
- NVIDIA Triton Inference Server
- OpenAI Whisper
- TensorRT-LLM

Please refer to their respective licenses for terms of use.
