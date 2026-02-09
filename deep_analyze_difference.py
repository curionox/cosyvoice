#!/usr/bin/env python3
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice
import torch
import numpy as np
import librosa

def compare_model_loading():
    """对比不同加载方式的模型差异"""
    print("=== 对比模型加载方式 ===")
    
    # WebUI方式加载（没有额外参数）
    print("1. WebUI方式加载:")
    model_webui = CosyVoice('pretrained_models/CosyVoice-300M-SFT')
    print(f"   load_jit: {getattr(model_webui, 'load_jit', '未设置')}")
    print(f"   fp16: {getattr(model_webui, 'fp16', '未设置')}")
    
    # API方式加载（显式参数）  
    print("2. API方式加载:")
    model_api = CosyVoice('pretrained_models/CosyVoice-300M-SFT', load_jit=False, fp16=False)
    print(f"   load_jit: {getattr(model_api, 'load_jit', '未设置')}")
    print(f"   fp16: {getattr(model_api, 'fp16', '未设置')}")
    
    # 测试相同角色生成差异
    text = "这是一个测试语音合成效果的句子"
    role = "真理医生"
    
    print(f"\n=== 测试角色: {role} ===")
    
    # WebUI方式调用
    print("WebUI方式调用:")
    webui_audio_list = []
    for i in model_webui.inference_sft(text, role, stream=False, speed=1.0):
        webui_audio_list.append(i['tts_speech'])
        print(f"   音频片段形状: {i['tts_speech'].shape}")
    
    if webui_audio_list:
        webui_audio = torch.cat(webui_audio_list, dim=1)
        webui_rms = torch.sqrt(torch.mean(webui_audio ** 2)).item()
        webui_max = webui_audio.abs().max().item()
        print(f"   最终RMS: {webui_rms:.6f}")
        print(f"   最大值: {webui_max:.6f}")
        print(f"   形状: {webui_audio.shape}")
    
    # API方式调用
    print("\nAPI方式调用:")
    api_audio_list = []
    for i in model_api.inference_sft(text, role, stream=False, speed=1.0):
        api_audio_list.append(i['tts_speech'])
        print(f"   音频片段形状: {i['tts_speech'].shape}")
    
    if api_audio_list:
        api_audio = torch.cat(api_audio_list, dim=1)
        api_rms = torch.sqrt(torch.mean(api_audio ** 2)).item()
        api_max = api_audio.abs().max().item()
        print(f"   最终RMS: {api_rms:.6f}")
        print(f"   最大值: {api_max:.6f}")
        print(f"   形状: {api_audio.shape}")
        
    # 比较两种方式的差异
    if webui_audio_list and api_audio_list:
        print("\n=== 两种加载方式的差异 ===")
        rms_diff = abs(webui_rms - api_rms)
        max_diff = abs(webui_max - api_max)
        print(f"RMS差异: {rms_diff:.6f}")
        print(f"最大值差异: {max_diff:.6f}")
        
        # 计算音频内容相似度
        min_len = min(webui_audio.shape[1], api_audio.shape[1])
        webui_trimmed = webui_audio[:, :min_len]
        api_trimmed = api_audio[:, :min_len]
        
        correlation = torch.corrcoef(torch.stack([webui_trimmed.flatten(), api_trimmed.flatten()]))[0,1].item()
        print(f"音频相关性: {correlation:.6f}")
        
        if correlation > 0.95:
            print("结论: 两种加载方式产生几乎相同的音频")
        else:
            print("结论: 两种加载方式产生不同的音频！")

def test_different_roles_webui_way():
    """用WebUI的方式测试不同角色"""
    print("\n=== WebUI方式测试不同角色 ===")
    
    model = CosyVoice('pretrained_models/CosyVoice-300M-SFT')
    text = "这是一个测试语音合成效果的句子"
    roles = ["真理医生", "林尼"]
    
    results = []
    
    for role in roles:
        print(f"\n测试角色: {role}")
        audio_list = []
        for i in model.inference_sft(text, role, stream=False, speed=1.0):
            audio_list.append(i['tts_speech'])
        
        if audio_list:
            audio = torch.cat(audio_list, dim=1)
            rms = torch.sqrt(torch.mean(audio ** 2)).item()
            
            # 计算更多特征
            audio_np = audio.numpy().flatten()
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_np, sr=22050)[0].mean()
            zcr = librosa.feature.zero_crossing_rate(audio_np)[0].mean()
            
            result = {
                'role': role,
                'rms': rms,
                'max': audio.abs().max().item(),
                'spectral_centroid': spectral_centroid,
                'zcr': zcr,
                'shape': audio.shape
            }
            results.append(result)
            
            print(f"   RMS: {rms:.6f}")
            print(f"   最大值: {result['max']:.6f}")
            print(f"   频谱质心: {spectral_centroid:.1f}")
            print(f"   过零率: {zcr:.6f}")
    
    if len(results) >= 2:
        print("\n=== WebUI方式角色差异 ===")
        rms_diff = abs(results[0]['rms'] - results[1]['rms'])
        centroid_diff = abs(results[0]['spectral_centroid'] - results[1]['spectral_centroid'])
        zcr_diff = abs(results[0]['zcr'] - results[1]['zcr'])
        
        print(f"RMS差异: {rms_diff:.6f}")
        print(f"频谱质心差异: {centroid_diff:.1f} Hz")
        print(f"过零率差异: {zcr_diff:.6f}")
        
        if rms_diff > 0.02 or centroid_diff > 200:
            print("结论: WebUI方式产生明显的角色差异")
        else:
            print("结论: WebUI方式角色差异较小")

if __name__ == "__main__":
    compare_model_loading()
    test_different_roles_webui_way()