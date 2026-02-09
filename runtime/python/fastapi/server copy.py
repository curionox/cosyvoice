# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import logging
import io
import random
import tempfile
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import torch
import torchaudio
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Insert Matcha-TTS first to avoid conflicts with root-level Matcha-TTS folder
sys.path.insert(0, '{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
sys.path.insert(0, '{}/../../..'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2, CosyVoice3, AutoModel
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


def generate_data(model_output):
    """Generate streaming audio data (raw PCM)"""
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


def generate_wav_data(model_output, sample_rate):
    """Generate complete WAV file data with proper headers"""
    audio_list = []
    for i in model_output:
        audio_list.append(i['tts_speech'])

    if not audio_list:
        return b""

    # Add 2 seconds of silence at the beginning
    silence_duration = 3.0  # seconds
    silence_samples = int(sample_rate * silence_duration)
    # Create silence tensor with same shape as audio [1, samples]
    silence = torch.zeros(1, silence_samples, dtype=audio_list[0].dtype, device=audio_list[0].device)

    # Insert silence at the beginning
    audio_list.insert(0, silence)

    # Concatenate all audio chunks (including silence)
    audio_data = torch.cat(audio_list, dim=1)

    # Use temporary file to avoid BytesIO issues with some backends
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # Save to temporary file
        torchaudio.save(temp_path, audio_data, sample_rate, backend='soundfile')

        # Read file content
        with open(temp_path, 'rb') as f:
            wav_data = f.read()

        return wav_data

    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(
    tts_text: str = Form(),
    spk_id: str = Form(),
    format: str = Form("stream"),
    speed: float = Form(1.0),
    stream: bool = Form(True),
    seed: int = Form(None)
):
    """SFT inference with audio format support"""
    # Handle seed
    if seed is None:
        seed = random.randint(1, 100000000)
    set_all_random_seed(seed)

    model_output = cosyvoice.inference_sft(tts_text, spk_id, stream=stream, speed=speed)

    if format == "wav":
        # Return complete WAV file
        wav_data = generate_wav_data(model_output, cosyvoice.sample_rate)
        return StreamingResponse(
            io.BytesIO(wav_data),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=tts_output.wav",
                "X-Seed-Used": str(seed)
            }
        )
    else:
        # Return streaming data
        return StreamingResponse(
            generate_data(model_output),
            media_type="audio/wav",
            headers={"X-Seed-Used": str(seed)}
        )


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(
    tts_text: str = Form(),
    prompt_text: str = Form(),
    prompt_wav: UploadFile = File(),
    format: str = Form("stream"),
    speed: float = Form(1.0),
    stream: bool = Form(True),
    seed: int = Form(None)
):
    """Zero-shot inference with audio format support"""
    # Handle seed
    if seed is None:
        seed = random.randint(1, 100000000)
    set_all_random_seed(seed)

    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed)

    if format == "wav":
        wav_data = generate_wav_data(model_output, cosyvoice.sample_rate)
        return StreamingResponse(
            io.BytesIO(wav_data),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=tts_output.wav",
                "X-Seed-Used": str(seed)
            }
        )
    else:
        return StreamingResponse(
            generate_data(model_output),
            media_type="audio/wav",
            headers={"X-Seed-Used": str(seed)}
        )


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(
    tts_text: str = Form(),
    prompt_wav: UploadFile = File(),
    format: str = Form("stream"),
    speed: float = Form(1.0),
    stream: bool = Form(True),
    seed: int = Form(None)
):
    """Cross-lingual inference with audio format support"""
    # Handle seed
    if seed is None:
        seed = random.randint(1, 100000000)
    set_all_random_seed(seed)

    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed)

    if format == "wav":
        wav_data = generate_wav_data(model_output, cosyvoice.sample_rate)
        return StreamingResponse(
            io.BytesIO(wav_data),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=tts_output.wav",
                "X-Seed-Used": str(seed)
            }
        )
    else:
        return StreamingResponse(
            generate_data(model_output),
            media_type="audio/wav",
            headers={"X-Seed-Used": str(seed)}
        )


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(
    tts_text: str = Form(),
    spk_id: str = Form(),
    instruct_text: str = Form(),
    format: str = Form("stream"),
    speed: float = Form(1.0),
    stream: bool = Form(True),
    seed: int = Form(None)
):
    """Instruct inference with audio format support"""
    # Handle seed
    if seed is None:
        seed = random.randint(1, 100000000)
    set_all_random_seed(seed)

    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text, stream=stream, speed=speed)

    if format == "wav":
        wav_data = generate_wav_data(model_output, cosyvoice.sample_rate)
        return StreamingResponse(
            io.BytesIO(wav_data),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=tts_output.wav",
                "X-Seed-Used": str(seed)
            }
        )
    else:
        return StreamingResponse(
            generate_data(model_output),
            media_type="audio/wav",
            headers={"X-Seed-Used": str(seed)}
        )


@app.get("/inference_instruct_sft")
@app.post("/inference_instruct_sft")
async def inference_instruct_sft(
    tts_text: str = Form(),
    spk_id: str = Form(),
    instruct_text: str = Form(""),
    format: str = Form("stream"),
    speed: float = Form(1.0),
    stream: bool = Form(True),
    seed: int = Form(None)
):
    """Use spk_id for voice (from spk2info.pt), optional instruct_text for style control. No audio upload needed.

    Supports both v1 spk2info (embedding-only) and v3 spk2info (full zero-shot data):
    - v3 spk2info with instruct_text: uses instruct2 path with cached speaker features + style control
    - v3 spk2info without instruct_text: uses zero_shot path with cached speaker features
    - v1 spk2info with instruct_text: uses SFT-instruct path (embedding + instruct text)
    - v1 spk2info without instruct_text: uses SFT path (embedding only)
    """
    if seed is None:
        seed = random.randint(1, 100000000)
    set_all_random_seed(seed)

    has_instruct = instruct_text and instruct_text.strip()
    is_v3 = cosyvoice.frontend._is_v3_spk2info(spk_id)

    if has_instruct:
        # With instruct_text: use inference_instruct_sft which handles both v1/v3 internally
        model_output = cosyvoice.inference_instruct_sft(tts_text, spk_id, instruct_text, stream=stream, speed=speed)
    elif is_v3:
        # No instruct, v3 spk2info: use zero_shot with cached speaker data
        model_output = cosyvoice.inference_zero_shot(tts_text, '', None, zero_shot_spk_id=spk_id, stream=stream, speed=speed)
    else:
        # No instruct, v1 spk2info: use plain SFT
        model_output = cosyvoice.inference_sft(tts_text, spk_id, stream=stream, speed=speed)

    if format == "wav":
        wav_data = generate_wav_data(model_output, cosyvoice.sample_rate)
        return StreamingResponse(
            io.BytesIO(wav_data),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=tts_output.wav",
                "X-Seed-Used": str(seed)
            }
        )
    else:
        return StreamingResponse(
            generate_data(model_output),
            media_type="audio/wav",
            headers={"X-Seed-Used": str(seed)}
        )


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(
    tts_text: str = Form(),
    instruct_text: str = Form(),
    prompt_wav: UploadFile = File(),
    format: str = Form("stream"),
    speed: float = Form(1.0),
    stream: bool = Form(True),
    seed: int = Form(None)
):
    """Instruct2 inference with audio format support"""
    # Handle seed
    if seed is None:
        seed = random.randint(1, 100000000)
    set_all_random_seed(seed)

    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k, stream=stream, speed=speed)

    if format == "wav":
        wav_data = generate_wav_data(model_output, cosyvoice.sample_rate)
        return StreamingResponse(
            io.BytesIO(wav_data),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=tts_output.wav",
                "X-Seed-Used": str(seed)
            }
        )
    else:
        return StreamingResponse(
            generate_data(model_output),
            media_type="audio/wav",
            headers={"X-Seed-Used": str(seed)}
        )


@app.get("/")
async def root():
    """API information"""
    return {
        "service": "CosyVoice FastAPI Server",
        "version": "2.1.0",
        "features": [
            "WAV format support (complete audio files with headers)",
            "Streaming audio support (raw PCM)",
            "Speed control",
            "Random seed support for reproducibility",
            "Multiple inference modes (SFT, Zero-shot, Cross-lingual, Instruct)",
            "v1/v3 spk2info auto-detection and compatibility"
        ],
        "endpoints": [
            "/inference_sft - SFT mode inference",
            "/inference_zero_shot - Zero-shot voice cloning",
            "/inference_cross_lingual - Cross-lingual synthesis",
            "/inference_instruct - Instruct-based synthesis (CosyVoice1 only)",
            "/inference_instruct_sft - Instruct with spk_id (no audio, supports v1+v3 spk2info)",
            "/inference_instruct2 - Instruct2-based synthesis (CosyVoice2/3)",
            "/list_speakers - List available speakers with format info",
            "/docs - API documentation"
        ],
        "parameters": {
            "format": "Output format: 'stream' (default, raw PCM) or 'wav' (complete WAV file)",
            "speed": "Speech speed multiplier (default: 1.0)",
            "stream": "Enable streaming generation (default: true)",
            "seed": "Random seed for reproducibility (optional, auto-generated if not provided)"
        }
    }


@app.get("/list_speakers")
async def list_speakers():
    """List available speakers with their spk2info format info"""
    speakers = []
    for spk_id in cosyvoice.list_available_spks():
        is_v3 = cosyvoice.frontend._is_v3_spk2info(spk_id)
        info = cosyvoice.frontend.spk2info[spk_id]
        speakers.append({
            "spk_id": spk_id,
            "format": "v3_zero_shot" if is_v3 else "v1_embedding",
            "keys": list(info.keys()),
            "supported_modes": (
                ["inference_sft", "inference_zero_shot", "inference_cross_lingual", "inference_instruct_sft", "inference_instruct2"]
                if is_v3 else
                ["inference_sft", "inference_instruct_sft"]
            )
        })
    model_type = cosyvoice.__class__.__name__
    return {
        "model_type": model_type,
        "sample_rate": cosyvoice.sample_rate,
        "total_speakers": len(speakers),
        "speakers": speakers
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=9234,
                        help='Server port')
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice-300M',
                        help='local path or modelscope repo id')
    parser.add_argument('--load_jit',
                        action='store_true',
                        help='Enable JIT compilation acceleration')
    parser.add_argument('--load_trt',
                        action='store_true',
                        help='Enable TensorRT acceleration')
    parser.add_argument('--fp16',
                        action='store_true',
                        help='Enable half precision (FP16)')
    parser.add_argument('--trt_concurrent',
                        type=int,
                        default=1,
                        help='TensorRT concurrent contexts')
    args = parser.parse_args()

    print(f"Loading model from: {args.model_dir}")
    print(f"Acceleration options - TRT: {args.load_trt}, JIT: {args.load_jit}, FP16: {args.fp16}")
    if args.load_trt and args.trt_concurrent > 1:
        print(f"TensorRT concurrent optimization: {args.trt_concurrent} contexts")

    # Try to load model with acceleration parameters
    # Detect model type by yaml file, then load with appropriate class
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        from modelscope import snapshot_download
        model_dir = snapshot_download(model_dir)

    if os.path.exists(os.path.join(model_dir, 'cosyvoice3.yaml')):
        # CosyVoice3: no load_jit param, only load_trt/fp16/load_vllm
        print("Detected CosyVoice3 model (cosyvoice3.yaml)")
        cosyvoice = CosyVoice3(
            model_dir,
            load_trt=args.load_trt,
            fp16=args.fp16,
            trt_concurrent=args.trt_concurrent
        )
        print(f"[OK] Model loaded as CosyVoice3 (TRT: {args.load_trt}, FP16: {args.fp16})")
    elif os.path.exists(os.path.join(model_dir, 'cosyvoice2.yaml')):
        print("Detected CosyVoice2 model (cosyvoice2.yaml)")
        cosyvoice = CosyVoice2(
            model_dir,
            load_jit=args.load_jit,
            load_trt=args.load_trt,
            fp16=args.fp16,
            trt_concurrent=args.trt_concurrent
        )
        print(f"[OK] Model loaded as CosyVoice2 (TRT: {args.load_trt}, JIT: {args.load_jit}, FP16: {args.fp16})")
    elif os.path.exists(os.path.join(model_dir, 'cosyvoice.yaml')):
        print("Detected CosyVoice model (cosyvoice.yaml)")
        cosyvoice = CosyVoice(
            model_dir,
            load_jit=args.load_jit,
            load_trt=args.load_trt,
            fp16=args.fp16,
            trt_concurrent=args.trt_concurrent
        )
        print(f"[OK] Model loaded as CosyVoice (TRT: {args.load_trt}, JIT: {args.load_jit}, FP16: {args.fp16})")
    else:
        raise TypeError(f'No valid model yaml found in {model_dir}! Expected cosyvoice.yaml, cosyvoice2.yaml, or cosyvoice3.yaml')

    print(f"\nStarting FastAPI server on port {args.port}")
    print(f"API Documentation: http://0.0.0.0:{args.port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
