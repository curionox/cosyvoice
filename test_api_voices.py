#!/usr/bin/env python3
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice
import torch
import librosa

def test_different_voices():
    # 加载模型
    model = CosyVoice('pretrained_models/CosyVoice-300M-SFT')
    spks = model.list_available_spks()
    
    # 测试文本
    text = "这是一个测试语音合成的句子"
    
    # 测试前3个不同音色
    test_spks = spks[:3]
    print(f"测试音色: {test_spks}")
    
    results = []
    for spk in test_spks:
        print(f"生成音色: {spk}")
        
        # 生成音频
        audio_list = []
        for i in model.inference_sft(text, spk, stream=False, speed=1.0):
            audio_list.append(i['tts_speech'])
        
        if audio_list:
            audio_data = torch.cat(audio_list, dim=1)
            # 计算音频特征用于比较
            audio_numpy = audio_data.numpy().flatten()
            rms = torch.sqrt(torch.mean(audio_data ** 2)).item()
            duration = len(audio_numpy) / model.sample_rate
            
            # 计算频谱质心作为声音特征
            stft = torch.stft(audio_data.squeeze(), n_fft=1024, hop_length=256, return_complex=True)
            magnitude = torch.abs(stft)
            freqs = torch.linspace(0, model.sample_rate/2, magnitude.size(0))
            spectral_centroid = torch.sum(freqs.unsqueeze(1) * magnitude, dim=0) / torch.sum(magnitude, dim=0)
            avg_centroid = torch.mean(spectral_centroid).item()
            
            results.append({
                'spk': spk,
                'rms': rms,
                'duration': duration,
                'spectral_centroid': avg_centroid
            })
            
            print(f"  RMS: {rms:.6f}, 时长: {duration:.2f}s, 频谱质心: {avg_centroid:.1f}Hz")
    
    # 比较结果
    print("\n=== 结果分析 ===")
    if len(results) >= 2:
        rms_diff = abs(results[0]['rms'] - results[1]['rms'])
        centroid_diff = abs(results[0]['spectral_centroid'] - results[1]['spectral_centroid'])
        
        print(f"RMS差异: {rms_diff:.6f}")
        print(f"频谱质心差异: {centroid_diff:.1f}Hz")
        
        if rms_diff < 0.001 and centroid_diff < 50:
            print("⚠️  警告: 不同音色生成的音频过于相似!")
        else:
            print("✅ 不同音色生成了不同的音频特征")
    
    return results

if __name__ == "__main__":
    results = test_different_voices()