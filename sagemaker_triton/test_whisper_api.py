#test_whisper_api.py
import requests
import base64
import json
import argparse

import os
import time
from datetime import datetime
import numpy as np
import logging
import requests
import base64
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import soundfile as sf
import io
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, parse_obj_as
import asyncio
import aiohttp
from fastapi import FastAPI, Request
import httpx
from fastapi.responses import StreamingResponse
from concurrent.futures import ThreadPoolExecutor

def encode_audio(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')

def test_ping(base_url):
    response = requests.get(f"{base_url}/ping")
    print("Ping Response:")
    print(json.dumps(response.json(), indent=2))
    print(f"Status Code: {response.status_code}")
    print()

def test_transcription(base_url, audio_file_path, language="", repo_id="whisper-large-v3", decoding_method="greedy_search", whisper_prompt="",repetition_penalty=1):
    encoded_audio = encode_audio(audio_file_path)
    
    payload = {
        "language": language,
        "repo_id": repo_id,
        "decoding_method": decoding_method,
        "whisper_prompt": whisper_prompt,
        "audio_data": encoded_audio,
        "repetition_penalty": repetition_penalty
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(f"{base_url}/invocations", json=payload, headers=headers)
    
    print("Transcription Response:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    print(f"Status Code: {response.status_code}")



server_url = "127.0.0.1:30005"
triton_client = httpclient.InferenceServerClient(url=server_url, verbose=False)

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 创建一个线程池来管理Triton客户端连接
# triton_client_pool = ThreadPoolExecutor(max_workers=10)

# def get_triton_client():
#     return asyncio.get_event_loop().run_in_executor(
#         triton_client_pool, 
#         lambda: httpclient.InferenceServerClient(url=server_url, verbose=False)
#     )

def send_whisper(whisper_prompt, audio_data, model_name, padding_duration=15):
    start_time = time.time()
    
    # Decode base64 audio data
    audio_bytes = base64.b64decode(audio_data)
    
    # Load audio from bytes
    with io.BytesIO(audio_bytes) as audio_file:
        waveform, sample_rate = sf.read(audio_file)
    
    assert sample_rate == 16000, f"Only support 16k sample rate, but got {sample_rate}"
    duration = int(len(waveform) / sample_rate)

    # padding to nearest 10 seconds
    samples = np.zeros(
        (
            1,
            padding_duration * sample_rate * ((duration // padding_duration) + 1),
        ),
        dtype=np.float32,
    )

    samples[0, : len(waveform)] = waveform

    inputs = [
        httpclient.InferInput(
            "WAV", samples.shape, np_to_triton_dtype(samples.dtype)
        ),
        httpclient.InferInput(
            "TEXT_PREFIX", [1, 1], "BYTES"
        ),
    ]
    inputs[0].set_data_from_numpy(samples)

    input_data_numpy = np.array([whisper_prompt], dtype=object)
    input_data_numpy = input_data_numpy.reshape((1, 1))
    inputs[1].set_data_from_numpy(input_data_numpy)

    outputs = [httpclient.InferRequestedOutput("TRANSCRIPTS")]
    sequence_id = np.random.randint(0, 1000000)

    inference_start_time = time.time()
    response = triton_client.infer(
        model_name, inputs, request_id=str(sequence_id), outputs=outputs
    )
    inference_end_time = time.time()

    decoding_results = response.as_numpy("TRANSCRIPTS")[0]
    if type(decoding_results) == np.ndarray:
        decoding_results = b" ".join(decoding_results).decode("utf-8")
    else:
        decoding_results = decoding_results.decode("utf-8")

    end_time = time.time()
    logging.info(f"Whisper processing time: {end_time - start_time:.3f} seconds")
    logging.info(f"Inference time: {inference_end_time - inference_start_time:.3f} seconds")
    return decoding_results, duration

def process(
    language: str,
    repo_id: str,
    decoding_method: str,
    whisper_prompt: str,
    audio_data: str,
):
    overall_start_time = time.time()

    logging.info(f"language: {language}")
    logging.info(f"repo_id: {repo_id}")
    logging.info(f"decoding_method: {decoding_method}")
    logging.info(f"whisper_prompt: {whisper_prompt}")

    model_name = "whisper"

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    logging.info(f"Started at {date_time}")

    whisper_start_time = time.time()
    text, duration = send_whisper(whisper_prompt, audio_data, model_name)
    whisper_end_time = time.time()

    date_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    overall_end_time = time.time()

    rtf = (overall_end_time - overall_start_time) / duration

    logging.info(f"Finished at {date_time}")
    logging.info(f"Total processing time: {overall_end_time - overall_start_time:.3f} seconds")
    logging.info(f"Whisper processing time: {whisper_end_time - whisper_start_time:.3f} seconds")

    info = f"""
    Wave duration  : {duration:.3f} s
    Processing time: {overall_end_time - overall_start_time:.3f} s
    RTF: {overall_end_time - overall_start_time:.3f}/{duration:.3f} = {rtf:.3f}
    """
    if rtf > 1:
        info += (
            "We are loading the model for the first run. "
            "Please run again to measure the real RTF."
        )

    logging.info(info)
    logging.info(f"\nrepo_id: {repo_id}\nhyp: {text}")

    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Whisper API")
    parser.add_argument("--url", type=str, default="http://localhost:8080", help="Base URL of the Whisper API")
    parser.add_argument("--audio", type=str, required=True, help="Path to the audio file")
    parser.add_argument("--language", type=str, default="", help="Language of the audio (optional)")
    parser.add_argument("--repo_id", type=str, default="whisper-large-v3", help="Model repository ID")
    parser.add_argument("--decoding_method", type=str, default="greedy_search", help="Decoding method")
    parser.add_argument("--prompt", type=str, default="", help="Whisper prompt (optional)")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    
    args = parser.parse_args()
    args.url ="http://127.0.0.1:30005"
    print("Testing Ping...")
    test_ping("http://127.0.0.1:30005")

    
    print("Testing Transcription...")
    encoded_audio = encode_audio(args.audio)
    # args.prompt = "<|startofprev|>This is MOBA game text<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>"
    # text = process(args.language, args.repo_id, args.decoding_method, args.prompt, encoded_audio)
    test_transcription(args.url, args.audio, args.language, args.repo_id, args.decoding_method, args.prompt, args.repetition_penalty)