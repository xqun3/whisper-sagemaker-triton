import os
import time
from datetime import datetime
import numpy as np
import logging
import uvicorn
import argparse
import requests
import tritonclient
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

import soundfile
import boto3
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, UploadFile, File

from pydantic import BaseModel, Field, parse_obj_as

from inference import *


server_url="127.0.0.1:8001"
triton_client = grpcclient.InferenceServerClient(url=server_url, verbose=False)
protocol_client = grpcclient

app = FastAPI()

logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('uvicorn')
    


def is_wav_file(filename):
    # 检查文件扩展名
    if not filename.lower().endswith('.wav'):
        return False
    
    # 检查文件是否存在
    if not os.path.isfile(filename):
        return False
    
    # 检查文件头
    try:
        with open(filename, 'rb') as file:
            header = file.read(12)
            if header[:4] != b'RIFF' or header[8:] != b'WAVE':
                return False
    except IOError:
        return False
    
    return True

def convert_to_wav(in_filename: str) -> str:
    """Convert the input audio file to a wave file"""
    start_time = time.time()
    
    if is_wav_file(in_filename):
        print(f"{in_filename} 是一个有效的 WAV 文件")
    else:
        print(f"{in_filename} 不是一个有效的 WAV 文件")
    
    out_filename = in_filename + ".wav"
    if '.mp3' in in_filename:
        _ = os.system(f"ffmpeg -y -i '{in_filename}' -acodec pcm_s16le -ac 1 -ar 16000 '{out_filename}' || exit 1")
    else:
        _ = os.system(f"ffmpeg -hide_banner -y -i '{in_filename}' -ar 16000 '{out_filename}' || exit 1")
    
    end_time = time.time()
    logging.info(f"Audio conversion time: {end_time - start_time:.3f} seconds")
    return out_filename

def load_audio(wav_path):
    start_time = time.time()
    waveform, sample_rate = soundfile.read(wav_path)
    assert sample_rate == 16000, f"Only support 16k sample rate, but got {sample_rate}"
    end_time = time.time()
    logging.info(f"Audio loading time: {end_time - start_time:.3f} seconds")
    return waveform, sample_rate

def send_whisper(whisper_prompt, wav_path, model_name, triton_client, protocol_client, padding_duration=10):
    start_time = time.time()
    
    waveform, sample_rate = soundfile.read(wav_path)
    assert sample_rate == 16000, f"Only support 16k sample rate, but got {sample_rate}"
    duration = int(len(waveform) / sample_rate)

    # padding to nearset 10 seconds
    samples = np.zeros(
        (
            1,
            padding_duration * sample_rate * ((duration // padding_duration) + 1),
        ),
        dtype=np.float32,
    )

    samples[0, : len(waveform)] = waveform

    lengths = np.array([[len(waveform)]], dtype=np.int32)

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
    # generate a random sequence id
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
        # For wenet
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
    s3_path: str,
):
    overall_start_time = time.time()
    
    logging.info(f"language: {language}")
    logging.info(f"repo_id: {repo_id}")
    logging.info(f"decoding_method: {decoding_method}")
    logging.info(f"whisper_prompt: {whisper_prompt}")
    logging.info(f"s3_path: {s3_path}")

    model_name = "whisper"
    
    download_start_time = time.time()
    in_filename = pre_download(s3_path)
    download_end_time = time.time()
    logging.info(f"Download time: {download_end_time - download_start_time:.3f} seconds")
    
    filename = convert_to_wav(in_filename)

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    logging.info(f"Started at {date_time}")

    whisper_start_time = time.time()
    text, duration = send_whisper(whisper_prompt, filename, model_name, triton_client, protocol_client) 
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


def download_from_s3(source_s3_url,local_file_path):
    s3 = boto3.client('s3')
    bucket_name, s3_file_path = get_bucket_and_key(source_s3_url)
    # 下载文件
    try:
        s3.download_file(bucket_name, s3_file_path, local_file_path)
        print(f'文件 {s3_file_path} 已下载到 {local_file_path}')
    except Exception as e:
        print(f'下载失败: {e}')

def pre_download(ref_wav_path:str)-> None:
    if "s3" in ref_wav_path:
        file_name = os.path.basename(ref_wav_path)
        download_file = "/tmp/"+file_name
        download_from_s3(ref_wav_path,download_file)
        return download_file
    else:
        return ref_wav_path

# def handle(
#         language: str="",
#         repo_id: str="whisper-large-v3",
#         decoding_method: str = "greedy_search",
#         whisper_prompt_textbox: str="",
#         s3_path: str=""
#         ):
#     if (
#             refer_wav_path == "" or refer_wav_path is None
#             or prompt_text == "" or prompt_text is None
#             or prompt_language == "" or prompt_language is None
#     ):
#         refer_wav_path, prompt_text, prompt_language = (
#             default_refer.path,
#             default_refer.text,
#             default_refer.language,
#         )
#         if not default_refer.is_ready():
#             return JSONResponse({"code": 400, "message": "未指定参考音频且接口无预设"}, status_code=400)


@app.get("/ping")
async def ping():
    """
    ping /ping func
    curl -v localhost:8000/v2/health/ready
    """
    # return {"message": "ok"}
    url = "http://127.0.0.1:10086/v2/health/ready"
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        print("Triton Inference Server is healthy and ready.")
        return JSONResponse({"code": 0, "message": "Success"}, status_code=200)
    else:
        print(f"Triton Inference Server is not ready. Status code: {response.status_code}")
        return JSONResponse({"code": 1, "message": "Fail"}, status_code=500)

@app.post("/invocations")
async def invocations(request: Request):
    json_post_raw = await request.json()
    print(f"invocations {json_post_raw=}")
    opt = parse_obj_as(InferenceOpt,json_post_raw)
    print(f"invocations {opt=}")
    text = process(opt.language, opt.repo_id, opt.decoding_method, opt.whisper_prompt, opt.s3_path)
    return JSONResponse({"code": 0, "message": "Success", "transcribe_text": text}, status_code=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="whisper api")
    parser.add_argument("-a", "--bind_addr", type=str, default="0.0.0.0", help="default: 0.0.0.0")
    parser.add_argument("-p", "--port", type=int, default="8080", help="default: 8080")
    args = parser.parse_args()

    uvicorn.run(app, host=args.bind_addr, port=args.port, workers=1, log_level="info")

