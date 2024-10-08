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
from pydub import AudioSegment

# def encode_audio(audio_file_path):
#     with open(audio_file_path, "rb") as audio_file:
#         return base64.b64encode(audio_file.read()).decode('utf-8')

def encode_audio(audio_file_path):
    # 加载音频文件
    audio = AudioSegment.from_wav(audio_file_path)
    
    # 检查是否为双通道
    if audio.channels == 2:
        print("检测到双通道音频，正在转换为单通道...")
        # 将双通道转换为单通道
        audio = audio.set_channels(1)
    
    # 将音频数据写入内存缓冲区
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)
    
    # 将缓冲区的内容编码为 base64
    return base64.b64encode(buffer.read()).decode('utf-8')

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