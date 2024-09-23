import boto3
import json
import base64
import os
from pydub import AudioSegment

def convert_audio_to_wav(input_path, output_path):
    """Convert audio file to 16kHz mono WAV"""
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_path, format="wav")

def read_audio_file(file_path):
    """Read audio file and return base64 encoded string"""
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')

def invoke_sagemaker_endpoint(runtime_client, endpoint_name, audio_data, language="", repo_id="whisper-large-v3", decoding_method="greedy_search", whisper_prompt=""):
    """Invoke SageMaker endpoint with audio data"""
    payload = {
        "language": language,
        "repo_id": repo_id,
        "decoding_method": decoding_method,
        "whisper_prompt": whisper_prompt,
        "audio_data": audio_data
    }
    
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    
    result = json.loads(response['Body'].read().decode())
    return result

def transcribe_audio(audio_path, endpoint_name, language="", repo_id="whisper-large-v3", decoding_method="greedy_search", whisper_prompt=""):
    # Convert audio to 16kHz mono WAV if it's not already
    if not audio_path.lower().endswith('.wav'):
        print("Converting audio to 16kHz mono WAV...")
        wav_path = os.path.splitext(audio_path)[0] + "_converted.wav"
        convert_audio_to_wav(audio_path, wav_path)
    else:
        wav_path = audio_path

    # Read and encode the audio file
    print("Reading and encoding audio file...")
    audio_data = read_audio_file(wav_path)

    # Create a SageMaker runtime client
    runtime_client = boto3.client('sagemaker-runtime')

    # Invoke the SageMaker endpoint
    print(f"Invoking SageMaker endpoint: {endpoint_name}")
    result = invoke_sagemaker_endpoint(
        runtime_client,
        endpoint_name,
        audio_data,
        language,
        repo_id,
        decoding_method,
        whisper_prompt
    )

    return result

# Example usage
if __name__ == "__main__":
    # Set your parameters here
    endpoint_name = "<Your SageMaker endpoint_name>"
    audio_path = "./English_04.wav"
    language = "en"  # Optional: specify the language
    repo_id = "whisper-large-v3"  # Optional: change the model if needed
    decoding_method = "greedy_search"  # Optional: change the decoding method if needed
    whisper_prompt = ""  # Optional: add a prompt if needed

    # Call the function
    result = transcribe_audio(audio_path, endpoint_name, language, repo_id, decoding_method, whisper_prompt)

    # Print the result
    print("Transcription result:")
    print(result)
