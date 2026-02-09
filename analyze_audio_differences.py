#!/usr/bin/env python3
import librosa
import numpy as np

def analyze_audio_file(file_path):
    """分析单个音频文件的特征"""
    try:
        y, sr = librosa.load(file_path)
        
        # 基本信息
        duration = len(y) / sr
        file_size = len(open(file_path, 'rb').read())
        
        # 音频特征
        rms = librosa.feature.rms(y=y)[0].mean()
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0].mean()
        
        # MFCC特征
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # 频谱统计
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean()
        
        return {
            'file': file_path,
            'duration': duration,
            'file_size': file_size,
            'rms': rms,
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'zero_crossing_rate': zero_crossing_rate,
            'mfcc_mean': mfcc_mean,
            'energy': np.sum(y**2)
        }
    except Exception as e:
        print(f"分析 {file_path} 出错: {e}")
        return None

def compare_audio_files():
    """对比不同来源的音频文件"""
    
    files_to_analyze = [
        # WebUI生成的文件
        ("WebUI - 真理医生", "audio (5).wav"),
        ("WebUI - 林尼", "audio (6).wav"),
        # API生成的文件
        ("API - 真理医生", "test_真理医生.wav"),
        ("API - 林尼", "test_林尼.wav")
    ]
    
    results = []
    print("=== 音频文件特征分析 ===\n")
    
    for label, file_path in files_to_analyze:
        result = analyze_audio_file(file_path)
        if result:
            result['label'] = label
            results.append(result)
            
            print(f"{label}:")
            print(f"  文件大小: {result['file_size']:,} bytes")
            print(f"  时长: {result['duration']:.2f}s")
            print(f"  RMS能量: {result['rms']:.6f}")
            print(f"  频谱质心: {result['spectral_centroid']:.1f} Hz")
            print(f"  频谱滚降: {result['spectral_rolloff']:.1f} Hz")
            print(f"  过零率: {result['zero_crossing_rate']:.6f}")
            print(f"  总能量: {result['energy']:.6f}")
            print()
    
    # 对比分析
    if len(results) >= 4:
        print("=== 差异对比分析 ===")
        
        # WebUI内部对比
        webui_doctor = results[0]
        webui_linni = results[1]
        api_doctor = results[2]
        api_linni = results[3]
        
        print("1. WebUI生成的两个角色差异:")
        webui_rms_diff = abs(webui_doctor['rms'] - webui_linni['rms'])
        webui_centroid_diff = abs(webui_doctor['spectral_centroid'] - webui_linni['spectral_centroid'])
        webui_energy_diff = abs(webui_doctor['energy'] - webui_linni['energy'])
        webui_mfcc_dist = np.linalg.norm(webui_doctor['mfcc_mean'] - webui_linni['mfcc_mean'])
        
        print(f"  RMS差异: {webui_rms_diff:.6f}")
        print(f"  频谱质心差异: {webui_centroid_diff:.1f} Hz")
        print(f"  能量差异: {webui_energy_diff:.6f}")
        print(f"  MFCC距离: {webui_mfcc_dist:.4f}")
        print()
        
        print("2. API生成的两个角色差异:")
        api_rms_diff = abs(api_doctor['rms'] - api_linni['rms'])
        api_centroid_diff = abs(api_doctor['spectral_centroid'] - api_linni['spectral_centroid'])
        api_energy_diff = abs(api_doctor['energy'] - api_linni['energy'])
        api_mfcc_dist = np.linalg.norm(api_doctor['mfcc_mean'] - api_linni['mfcc_mean'])
        
        print(f"  RMS差异: {api_rms_diff:.6f}")
        print(f"  频谱质心差异: {api_centroid_diff:.1f} Hz")
        print(f"  能量差异: {api_energy_diff:.6f}")
        print(f"  MFCC距离: {api_mfcc_dist:.4f}")
        print()
        
        print("3. WebUI vs API 差异比较:")
        print(f"  WebUI角色差异程度: {webui_mfcc_dist:.4f}")
        print(f"  API角色差异程度: {api_mfcc_dist:.4f}")
        print(f"  差异比率: {webui_mfcc_dist / api_mfcc_dist:.2f}x")
        
        if webui_mfcc_dist > api_mfcc_dist * 1.5:
            print("  结论: WebUI生成的角色差异明显更大!")
        else:
            print("  结论: 两者差异程度相近")
            
        print()
        
        # 同角色对比
        print("4. 同角色在不同平台的差异:")
        doctor_mfcc_diff = np.linalg.norm(webui_doctor['mfcc_mean'] - api_doctor['mfcc_mean'])
        linni_mfcc_diff = np.linalg.norm(webui_linni['mfcc_mean'] - api_linni['mfcc_mean'])
        
        print(f"  真理医生 - WebUI vs API MFCC距离: {doctor_mfcc_diff:.4f}")
        print(f"  林尼 - WebUI vs API MFCC距离: {linni_mfcc_diff:.4f}")

if __name__ == "__main__":
    compare_audio_files()