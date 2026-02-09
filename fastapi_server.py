#!/usr/bin/env python3
import sys
import os
import argparse
import logging
import io
import random
#  python fastapi_server.py --model_dir pretrained_models/CosyVoice-300M-SFT --port 9234  --load_jit --fp16 --load_trt
# 添加必要的路径
sys.path.append('third_party/Matcha-TTS')

from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import StreamingResponse
from cosyvoice.utils.file_utils import load_wav
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.common import set_all_random_seed

app = FastAPI()

# 设置跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

def split_text_smartly(text, max_length=200):
    """智能文本分割，按句号、逗号、问号等标点符号分割"""
    if len(text) <= max_length:
        return [text]

    import re
    
    # 使用正确的中英文标点符号
    # 优先级：句号、问号、感叹号 > 分号 > 逗号
    sentence_endings = r'([。？！.?!])'
    major_pauses = r'([；;])'
    minor_pauses = r'([，,])'
    
    # 先尝试按句子结束符分割
    parts = re.split(sentence_endings, text)
    
    # 重组句子
    segments = []
    for i in range(0, len(parts), 2):
        if i < len(parts) - 1:
            segments.append(parts[i] + parts[i + 1])
        elif parts[i].strip():
            segments.append(parts[i])
    
    # 合并短句子，分割长句子
    chunks = []
    current_chunk = ""
    
    for segment in segments:
        if len(current_chunk) + len(segment) <= max_length:
            current_chunk += segment
        else:
            if current_chunk:
                chunks.append(current_chunk)
            
            # 如果单个句子太长，尝试进一步分割
            if len(segment) > max_length:
                # 按逗号再分割
                sub_parts = re.split(minor_pauses, segment)
                sub_chunk = ""
                for j in range(0, len(sub_parts), 2):
                    if j < len(sub_parts) - 1:
                        part = sub_parts[j] + sub_parts[j + 1]
                    else:
                        part = sub_parts[j]
                    
                    if len(sub_chunk) + len(part) <= max_length:
                        sub_chunk += part
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk)
                        sub_chunk = part
                
                if sub_chunk:
                    chunks.append(sub_chunk)
                current_chunk = ""
            else:
                current_chunk = segment
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return [chunk for chunk in chunks if chunk.strip()]

def normalize_audio_chunks(audio_chunks, target_db=-20):
    """标准化音频片段的音量"""
    import torch
    
    if not audio_chunks:
        return audio_chunks
    
    normalized_chunks = []
    print(f"开始标准化 {len(audio_chunks)} 个音频片段...")
    
    for i, chunk in enumerate(audio_chunks):
        # 计算当前音频的RMS
        rms = torch.sqrt(torch.mean(chunk ** 2))
        
        # 计算目标RMS（基于目标dB）
        target_rms = 10 ** (target_db / 20)
        
        # 计算缩放因子
        if rms > 0:
            scale = target_rms / rms
            normalized_chunk = chunk * scale
            print(f"  片段 {i+1}: RMS {rms:.6f} -> {target_rms:.6f}, 缩放 {scale:.3f}")
        else:
            normalized_chunk = chunk
            print(f"  片段 {i+1}: 静音片段，跳过标准化")
            
        # 防止削波
        max_val = torch.abs(normalized_chunk).max()
        if max_val > 0.95:
            clip_scale = 0.95 / max_val
            normalized_chunk = normalized_chunk * clip_scale
            print(f"    应用削波保护: {clip_scale:.3f}")
            
        normalized_chunks.append(normalized_chunk)
    
    print("音频标准化完成")
    return normalized_chunks

def generate_data(model_output):
    """将模型输出转换为流式音频数据"""
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio

def generate_wav_data(model_output, sample_rate, normalize_audio=True, target_db=-20):
    """将模型输出转换为完整WAV文件数据"""
    import torch
    import torchaudio
    import io
    import tempfile
    import os
    
    audio_list = []
    for i in model_output:
        audio_list.append(i['tts_speech'])
    
    if not audio_list:
        return b""
    
    # 如果启用音频标准化且有多个片段，进行标准化处理
    if normalize_audio and len(audio_list) > 1:
        print(f"检测到 {len(audio_list)} 个音频片段，启用音量标准化...")
        audio_list = normalize_audio_chunks(audio_list, target_db)
    elif normalize_audio and len(audio_list) == 1:
        print("单个音频片段，应用基础标准化...")
        # 对单个片段也进行基础标准化
        chunk = audio_list[0]
        max_val = torch.abs(chunk).max()
        if max_val > 0:
            # 标准化到0.8的峰值，留20%余量防止削波
            audio_list[0] = chunk * (0.8 / max_val)
            print(f"  单片段标准化: 峰值 {max_val:.3f} -> 0.8")
    
    # 合并音频片段
    audio_data = torch.cat(audio_list, dim=1)
    
    # 最终全局标准化
    if normalize_audio:
        final_max = torch.abs(audio_data).max()
        if final_max > 0.95:
            global_scale = 0.95 / final_max
            audio_data = audio_data * global_scale
            print(f"应用全局削波保护: {global_scale:.3f}")
    
    # 使用临时文件避免torchcodec的BytesIO问题
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # 保存到临时文件
        torchaudio.save(temp_path, audio_data, sample_rate, backend='soundfile')
        
        # 读取文件内容
        with open(temp_path, 'rb') as f:
            wav_data = f.read()
        
        return wav_data
    
    finally:
        # 清理临时文件
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
    max_text_length: int = Form(200),  # 提高默认值到200
    enable_smart_split: bool = Form(True),
    seed: int = Form(None),  # 新增seed参数
    normalize_audio: bool = Form(False),  # 新增音频标准化参数
    target_db: float = Form(-20.0)  # 新增目标音量参数
):
    """SFT模式推理，支持智能文本分割和流式合成"""
    import time
    start_time = time.time()
    
    # 种子处理：如果没有提供，自动生成一个
    if seed is None:
        seed = random.randint(1, 100000000)
        print(f"自动生成种子: {seed}")
    else:
        print(f"使用指定种子: {seed}")
    
    # 设置初始种子
    set_all_random_seed(seed)
    
    # 当format=wav时，强制使用非流式推理以保证音频质量
    # 流式推理的token分批处理可能导致音频内容与文本不匹配
    use_stream = stream if format != "wav" else False
    if format == "wav" and stream:
        print(f"WAV格式输出: 强制使用非流式推理以保证质量")

    # 智能文本分割优化 - 200字符以内不分割
    if len(tts_text) > max_text_length and enable_smart_split:
        print(f"长文本检测 ({len(tts_text)} 字符)，启用智能分割...")
        text_chunks = split_text_smartly(tts_text, max_text_length)
        print(f"分割为 {len(text_chunks)} 个片段")

        # 对每个片段进行合成
        def generate_chunks():
            for i, chunk in enumerate(text_chunks):
                print(f"合成片段 {i+1}/{len(text_chunks)}: {chunk[:30]}...")

                # 重要：每个片段使用相同的基础种子，但加上偏移量保证轻微变化
                # 这样既保证了一致性，又避免了完全相同的音频
                chunk_seed = seed + i
                set_all_random_seed(chunk_seed)

                chunk_output = cosyvoice.inference_sft(chunk, spk_id, stream=use_stream, speed=speed)
                for audio in chunk_output:
                    yield audio

        model_output = generate_chunks()
    else:
        # 单段文本合成，已经设置了种子
        print(f"短文本或禁用分割 ({len(tts_text)} 字符)，使用单次生成 (stream={use_stream})")
        model_output = cosyvoice.inference_sft(tts_text, spk_id, stream=use_stream, speed=speed)
    
    processing_time = time.time() - start_time
    text_length = len(tts_text)
    print(f"文本处理完成 - 长度: {text_length}, 耗时: {processing_time:.2f}s")
    
    if format == "wav":
        # 返回完整WAV文件
        wav_data = generate_wav_data(model_output, cosyvoice.sample_rate, normalize_audio, target_db)
        total_time = time.time() - start_time
        rtf = total_time / (len(wav_data) / cosyvoice.sample_rate / 2)  # 估算RTF
        print(f"WAV生成完成 - 总耗时: {total_time:.2f}s, 估算RTF: {rtf:.3f}")
        
        return StreamingResponse(
            io.BytesIO(wav_data), 
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=tts_output.wav",
                "X-Processing-Time": str(total_time),
                "X-RTF": str(rtf),
                "X-Seed-Used": str(seed),  # 返回使用的种子
                "X-Audio-Normalized": str(normalize_audio),  # 返回是否标准化
                "X-Target-DB": str(target_db) if normalize_audio else "N/A"  # 返回目标音量
            }
        )
    else:
        # 返回流式数据
        return StreamingResponse(
            generate_data(model_output), 
            media_type="audio/wav",
            headers={"X-Seed-Used": str(seed)}  # 返回使用的种子
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
    """带语气控制的语音合成

    关键优化（防止提示词泄露和重复）：
    1. instruct_text 自动添加 <|endofprompt|> 标记
    2. tts_text 前添加换行符物理隔离指令

    通过 instruct_text 参数指定语气，例如:
    - "用悲伤的语气说"
    - "用开心愉快的语气说"
    - "用愤怒的语气说"
    - "用温柔的语气说"
    - "Speak with a sad tone"

    参数:
        tts_text: 要合成的文本
        spk_id: 说话人ID
        instruct_text: 语气指令（如 "用悲伤的语气说"）
        format: 输出格式 stream/wav
        speed: 语速 (默认1.0)
        seed: 随机种子
    """
    import time
    start_time = time.time()

    if seed is None:
        seed = random.randint(1, 100000000)
        print(f"自动生成种子: {seed}")
    set_all_random_seed(seed)

    # 当format=wav时，强制使用非流式推理
    use_stream = stream if format != "wav" else False

    has_instruct = instruct_text and instruct_text.strip()

    print(f"Instruct SFT 推理 - 文本: {tts_text[:30]}..., 语气: {instruct_text}, spk_id: {spk_id}")

    if has_instruct:
        # --- 关键修复：格式化指令文本 ---
        clean_instruct = instruct_text.strip().replace('\n', ' ')
        if '<|endofprompt|>' not in clean_instruct:
            clean_instruct = f"{clean_instruct}<|endofprompt|>"

        # --- 关键修复：在文本前加换行符，物理隔离指令和文本 ---
        final_tts_text = f"\n{tts_text.strip()}"

        # 带语气控制的推理
        model_output = cosyvoice.inference_instruct_sft(final_tts_text, spk_id, clean_instruct, stream=use_stream, speed=speed)
    else:
        # 无语气指令，使用普通SFT
        model_output = cosyvoice.inference_sft(tts_text, spk_id, stream=use_stream, speed=speed)

    if format == "wav":
        wav_data = generate_wav_data(model_output, cosyvoice.sample_rate, normalize_audio=False)
        total_time = time.time() - start_time
        print(f"Instruct SFT WAV生成完成 - 耗时: {total_time:.2f}s")
        return StreamingResponse(
            io.BytesIO(wav_data),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=tts_instruct_output.wav",
                "X-Seed-Used": str(seed)
            }
        )
    else:
        return StreamingResponse(
            generate_data(model_output),
            media_type="audio/wav",
            headers={"X-Seed-Used": str(seed)}
        )

@app.post("/inference_emotion_clone")
async def inference_emotion_clone(
    tts_text: str = Form(),
    spk_id: str = Form(),
    emotion_audio: UploadFile = File(...),
    emotion_prompt: str = Form(""),
    format: str = Form("wav"),
    speed: float = Form(1.0),
    seed: int = Form(None)
):
    """情感克隆：保留音色 + 克隆情感

    用法：
    - spk_id: 你的音色ID（保留音色）
    - emotion_audio: 情感参考音频（克隆情感，可以是任何人的声音）
    - emotion_prompt: 情感描述（可选，如 "用愤怒的语气说"）

    示例：
    curl -X POST http://localhost:9234/inference_emotion_clone \
      -F "tts_text=你好世界" \
      -F "spk_id=17" \
      -F "emotion_audio=@angry_reference.wav" \
      -F "emotion_prompt=用愤怒的语气说" \
      -F "format=wav" \
      --output output.wav
    """
    import time
    start_time = time.time()

    if seed is None:
        seed = random.randint(1, 100000000)
    set_all_random_seed(seed)

    # 读取情感参考音频
    emotion_audio_data = await emotion_audio.read()
    emotion_audio_16k = load_wav(io.BytesIO(emotion_audio_data), 16000)

    # 构建情感指令
    clean_prompt = emotion_prompt.strip().replace('\n', ' ') if emotion_prompt else ""
    if clean_prompt:
        instruct = f"{clean_prompt}<|endofprompt|>"
    else:
        instruct = "Neutral<|endofprompt|>"

    # 在文本前加换行符，物理隔离指令和文本
    final_tts_text = f"\n{tts_text.strip()}"

    print(f"情感克隆推理 - 文本: {tts_text[:30]}..., 音色: {spk_id}, 情感指令: {clean_prompt}")

    # 使用 inference_instruct2：情感音频 + spk_id音色
    # 注意：这里用情感音频作为 prompt_wav，用 spk_id 保持音色一致性
    model_output = cosyvoice.inference_instruct2(
        final_tts_text,
        instruct,
        emotion_audio_16k,
        zero_shot_spk_id=spk_id,  # 使用已注册的音色
        stream=False,
        speed=speed
    )

    wav_data = generate_wav_data(model_output, cosyvoice.sample_rate, normalize_audio=False)
    total_time = time.time() - start_time
    print(f"情感克隆完成 - 耗时: {total_time:.2f}s")

    return StreamingResponse(
        io.BytesIO(wav_data),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=emotion_clone_output.wav",
            "X-Seed-Used": str(seed),
            "X-Processing-Time": str(total_time)
        }
    )

@app.post("/inference_zero_shot_emotion")
async def inference_zero_shot_emotion(
    tts_text: str = Form(),
    voice_audio: UploadFile = File(...),
    emotion_prompt: str = Form(""),
    format: str = Form("wav"),
    speed: float = Form(1.0),
    seed: int = Form(None)
):
    """零样本情感合成：上传音色音频 + 情感控制

    用法：
    - voice_audio: 音色参考音频（你的声音，用于克隆音色）
    - emotion_prompt: 情感描述（如 "用愤怒的语气说"）

    注意：这个接口音色和情感来自同一个音频，
    如果想要分离音色和情感，请用 /inference_emotion_clone
    """
    import time
    start_time = time.time()

    if seed is None:
        seed = random.randint(1, 100000000)
    set_all_random_seed(seed)

    # 读取音色音频
    voice_audio_data = await voice_audio.read()
    voice_audio_16k = load_wav(io.BytesIO(voice_audio_data), 16000)

    # 构建情感指令
    clean_prompt = emotion_prompt.strip().replace('\n', ' ') if emotion_prompt else ""
    if clean_prompt:
        instruct = f"{clean_prompt}<|endofprompt|>"
    else:
        instruct = "Neutral<|endofprompt|>"

    # 在文本前加换行符，物理隔离指令和文本
    final_tts_text = f"\n{tts_text.strip()}"

    print(f"零样本情感合成 - 文本: {tts_text[:30]}..., 情感指令: {clean_prompt}")

    model_output = cosyvoice.inference_instruct2(
        final_tts_text,
        instruct,
        voice_audio_16k,
        stream=False,
        speed=speed
    )

    wav_data = generate_wav_data(model_output, cosyvoice.sample_rate, normalize_audio=False)
    total_time = time.time() - start_time
    print(f"零样本情感合成完成 - 耗时: {total_time:.2f}s")

    return StreamingResponse(
        io.BytesIO(wav_data),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=zero_shot_emotion_output.wav",
            "X-Seed-Used": str(seed),
            "X-Processing-Time": str(total_time)
        }
    )

@app.get("/list_voices")
@app.get("/list_speakers")
async def list_voices():
    """获取可用音色列表，包含格式信息"""
    try:
        voices = cosyvoice.list_available_spks()
        # 检查每个speaker的格式
        detailed_voices = []
        for spk_id in voices:
            info = cosyvoice.frontend.spk2info.get(spk_id, {})
            v3_required_keys = {'llm_prompt_speech_token', 'flow_prompt_speech_token', 'prompt_speech_feat', 'llm_embedding', 'flow_embedding'}
            is_v3 = v3_required_keys.issubset(info.keys())
            detailed_voices.append({
                "spk_id": spk_id,
                "format": "v3_zero_shot" if is_v3 else "v1_embedding",
                "instruct_support": "强" if is_v3 else "弱"
            })
        return {
            "code": 0,
            "msg": "success",
            "model_type": cosyvoice.__class__.__name__,
            "voices": detailed_voices,
            "count": len(voices),
            "note": "v3_zero_shot格式支持更强的情感控制，v1_embedding格式情感控制较弱"
        }
        return {
            "code": 0,
            "msg": "success", 
            "voices": voices,
            "count": len(voices)
        }
    except Exception as e:
        return {"code": 1, "msg": str(e), "voices": [], "count": 0}

@app.post("/warmup_trt")
async def warmup_trt(
    sample_text: str = Form("你好，这是一个测试文本。"),
    spk_id: str = Form("中文女")
):
    """预热TensorRT引擎，生成优化的引擎文件"""
    if not hasattr(cosyvoice, 'load_trt') or not cosyvoice.load_trt:
        return {"code": 1, "msg": "TensorRT未启用，无法预热"}
    
    try:
        import time
        start_time = time.time()
        
        print("开始TensorRT引擎预热...")
        # 执行几次推理来触发TensorRT引擎编译和优化
        for i in range(3):
            print(f"预热轮次 {i+1}/3")
            output = list(cosyvoice.inference_sft(sample_text, spk_id, stream=False))
        
        warmup_time = time.time() - start_time
        print(f"TensorRT预热完成，耗时: {warmup_time:.2f}s")
        
        return {
            "code": 0,
            "msg": "TensorRT引擎预热完成",
            "warmup_time": warmup_time,
            "note": "后续推理将使用优化后的引擎，速度会显著提升"
        }
    except Exception as e:
        return {"code": 1, "msg": f"预热失败: {str(e)}"}

@app.get("/performance_info")
async def performance_info():
    """获取当前性能配置信息"""
    config = {
        "model_path": getattr(cosyvoice, 'model_dir', 'unknown'),
        "sample_rate": getattr(cosyvoice, 'sample_rate', 22050),
        "acceleration": {
            "tensorrt": getattr(cosyvoice, 'load_trt', False),
            "jit": getattr(cosyvoice, 'load_jit', False), 
            "fp16": getattr(cosyvoice, 'fp16', False),
            "trt_concurrent": getattr(cosyvoice, 'trt_concurrent', 1)
        },
        "optimization_tips": [
            "使用 --fp16 启用半精度加速 (30-50% 提速)",
            "使用 --load_trt 启用TensorRT加速",
            "使用 --load_jit 启用JIT编译加速", 
            "调用 /warmup_trt 预热TensorRT引擎",
            "长文本自动启用智能分割优化"
        ]
    }
    return config

@app.get("/")
async def root():
    """根路径信息"""
    return {
        "service": "CosyVoice FastAPI Server (优化版)",
        "version": "1.3.0",
        "features": [
            "FP16半精度加速",
            "TensorRT + JIT双重加速", 
            "智能文本分割 (修复标点符号)",
            "流式合成优化",
            "TensorRT引擎预热",
            "随机种子支持 (自动生成/手动指定)",
            "声音一致性保证",
            "音频标准化 (解决音量不一致问题)",
            "RMS音量标准化"
        ],
        "endpoints": [
            "/inference_sft - SFT模式语音合成 (支持智能分割、种子控制和音频标准化)",
            "/list_voices - 获取音色列表",
            "/warmup_trt - 预热TensorRT引擎",
            "/performance_info - 获取性能配置信息",
            "/docs - API文档"
        ],
        "new_parameters": {
            "seed": "随机种子 (可选，为空时自动生成)",
            "max_text_length": "文本分割阈值 (默认200字符)",
            "enable_smart_split": "是否启用智能分割 (默认true)",
            "normalize_audio": "是否启用音频标准化 (默认true)",
            "target_db": "目标音量dB值 (默认-20.0)"
        }
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8234, help='服务端口')
    parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice-300M-SFT', 
                        help='模型路径')
    parser.add_argument('--load_trt', action='store_true', help='启用TensorRT加速')
    parser.add_argument('--load_jit', action='store_true', help='启用JIT编译加速')
    parser.add_argument('--fp16', action='store_true', help='启用半精度浮点数')
    parser.add_argument('--trt_concurrent', type=int, default=4, help='TensorRT并发数')
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model_dir}")
    print(f"Acceleration options - TRT: {args.load_trt}, JIT: {args.load_jit}, FP16: {args.fp16}, TRT Concurrent: {args.trt_concurrent}")
    if args.load_trt and args.trt_concurrent > 1:
        print(f"⚡ TensorRT并发优化: {args.trt_concurrent}个上下文 (预期25%+性能提升)")
    
    # 加载模型（与官方方式相同）
    try:
        print("尝试加载 CosyVoice...")
        cosyvoice = CosyVoice(args.model_dir, 
                             load_jit=args.load_jit, 
                             load_trt=args.load_trt, 
                             fp16=args.fp16, 
                             trt_concurrent=args.trt_concurrent)
        print(f"✓ Model loaded as CosyVoice (TRT: {args.load_trt}, JIT: {args.load_jit}, FP16: {args.fp16})")
    except Exception as e1:
        print(f"✗ CosyVoice 加载失败: {e1}")
        print("尝试加载 CosyVoice2...")
        try:
            cosyvoice = CosyVoice2(args.model_dir, 
                                  load_jit=args.load_jit, 
                                  load_trt=args.load_trt, 
                                  fp16=args.fp16, 
                                  trt_concurrent=args.trt_concurrent)
            print(f"✓ Model loaded as CosyVoice2 (TRT: {args.load_trt}, JIT: {args.load_jit}, FP16: {args.fp16})")
        except Exception as e2:
            print(f"✗ CosyVoice2 加载失败: {e2}")
            print(f"\n详细错误信息:")
            print(f"CosyVoice 错误: {e1}")
            print(f"CosyVoice2 错误: {e2}")
            raise TypeError('No valid model type!')
    
    # 打印可用音色
    try:
        voices = cosyvoice.list_available_spks()
        print(f"Available voices ({len(voices)}): {voices[:5]}...")  # 显示前5个
    except Exception as e:
        print(f"Warning: Could not list voices: {e}")
    
    print(f"\nStarting FastAPI server on port {args.port}")
    print(f"API Documentation: http://0.0.0.0:{args.port}/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
