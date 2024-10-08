import os
import time
from datetime import datetime
import numpy as np
import logging
import uvicorn
import argparse
import requests
import base64
import tritonclient
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

import soundfile as sf
import io
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, parse_obj_as

server_url="127.0.0.1:8001"
triton_client = grpcclient.InferenceServerClient(url=server_url, verbose=False)
protocol_client = grpcclient

app = FastAPI()

logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('uvicorn')

class InferenceOpt(BaseModel):
    language: str = Field(default="")
    repo_id: str = Field(default="whisper-large-v3")
    decoding_method: str = Field(default="greedy_search")
    whisper_prompt: str = Field(default="")
    audio_data: str = Field(...)  # Base64 encoded audio data

def send_whisper(whisper_prompt, audio_data, model_name, triton_client, protocol_client, padding_duration=15):
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
        protocol_client.InferInput(
            "WAV", samples.shape, np_to_triton_dtype(samples.dtype)
        ),
        protocol_client.InferInput(
            "TEXT_PREFIX", [1, 1], "BYTES"
        ),
    ]
    inputs[0].set_data_from_numpy(samples)

    input_data_numpy = np.array([whisper_prompt], dtype=object)
    input_data_numpy = input_data_numpy.reshape((1, 1))
    inputs[1].set_data_from_numpy(input_data_numpy)

    outputs = [protocol_client.InferRequestedOutput("TRANSCRIPTS")]
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
    text, duration = send_whisper(whisper_prompt, audio_data, model_name, triton_client, protocol_client) 
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

@app.get("/ping")
async def ping():
    url = "http://127.0.0.1:10086/v2/health/ready"
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        # print("Triton Inference Server is healthy and ready.")
        return JSONResponse({"code": 0, "message": "Success"}, status_code=200)
    else:
        # print(f"Triton Inference Server is not ready. Status code: {response.status_code}")
        return JSONResponse({"code": 1, "message": "Fail"}, status_code=500)

@app.post("/invocations")
async def invocations(request: Request):
    json_post_raw = await request.json()
    # print(f"invocations {json_post_raw=}")
    opt = parse_obj_as(InferenceOpt, json_post_raw)
    # print(f"invocations {opt=}")
    text = process(opt.language, opt.repo_id, opt.decoding_method, opt.whisper_prompt, opt.audio_data)
    return JSONResponse({"code": 0, "message": "Success", "transcribe_text": text}, status_code=200)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="whisper api")
    parser.add_argument("-a", "--bind_addr", type=str, default="0.0.0.0", help="default: 0.0.0.0")
    parser.add_argument("-p", "--port", type=int, default="8080", help="default: 8080")
    args = parser.parse_args()

    uvicorn.run(app, host=args.bind_addr, port=args.port, workers=1, log_level="info")
