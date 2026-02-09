#!/usr/bin/env python3
"""
CosyVoice 性能基准测试脚本
测试不同配置下的推理速度和RTF
"""
import sys
import os
import argparse
import time
import statistics

# 添加必要的路径
sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2

def calculate_rtf(audio_duration, processing_time):
    """计算实时因子 (Real-Time Factor)"""
    return processing_time / audio_duration if audio_duration > 0 else float('inf')

def estimate_audio_duration(audio_data, sample_rate=22050):
    """估算音频时长"""
    if hasattr(audio_data, 'shape'):
        return audio_data.shape[-1] / sample_rate
    return 1.0  # 默认估算

def benchmark_config(model_dir, load_trt=False, load_jit=False, fp16=False, trt_concurrent=1):
    """测试特定配置的性能"""
    print(f"\n{'='*60}")
    print(f"测试配置: TRT={load_trt}, JIT={load_jit}, FP16={fp16}, Concurrent={trt_concurrent}")
    print(f"{'='*60}")
    
    # 加载模型
    start_time = time.time()
    try:
        cosyvoice = CosyVoice(model_dir, 
                             load_jit=load_jit, 
                             load_trt=load_trt, 
                             fp16=fp16, 
                             trt_concurrent=trt_concurrent)
        model_type = "CosyVoice"
    except Exception as e1:
        try:
            cosyvoice = CosyVoice2(model_dir, 
                                  load_jit=load_jit, 
                                  load_trt=load_trt, 
                                  fp16=fp16, 
                                  trt_concurrent=trt_concurrent)
            model_type = "CosyVoice2"
        except Exception as e2:
            print(f"模型加载失败: {e1}, {e2}")
            return None
    
    load_time = time.time() - start_time
    print(f"模型加载: {model_type}, 耗时: {load_time:.2f}s")
    
    # 获取音色
    try:
        voices = cosyvoice.list_available_spks()
        test_voice = voices[0] if voices else "中文女"
    except:
        test_voice = "中文女"
    
    # 测试文本
    test_cases = [
        ("短文本", "你好，世界！"),
        ("中文本", "这是一个中等长度的测试文本，用于评估语音合成的性能表现。"),
        ("长文本", "人工智能技术的发展日新月异，语音合成作为其中的重要分支，已经在各个领域得到了广泛的应用。从智能助手到有声读物，从导航系统到客服机器人，语音合成技术正在改变着我们与机器交互的方式。随着深度学习技术的不断进步，现代的语音合成系统能够生成更加自然、流畅的语音，为用户带来更好的体验。")
    ]
    
    results = []
    
    for case_name, text in test_cases:
        print(f"\n测试 {case_name} (长度: {len(text)})")
        
        # 多次测试取平均值
        times = []
        rtfs = []
        
        for i in range(3):
            start = time.time()
            try:
                output = list(cosyvoice.inference_sft(text, test_voice, stream=False))
                end = time.time()
                
                processing_time = end - start
                times.append(processing_time)
                
                # 估算音频时长和RTF
                if output:
                    audio_duration = estimate_audio_duration(output[0]['tts_speech'], cosyvoice.sample_rate)
                    rtf = calculate_rtf(audio_duration, processing_time)
                    rtfs.append(rtf)
                
                print(f"  轮次 {i+1}: {processing_time:.2f}s, RTF: {rtf:.3f}")
                
            except Exception as e:
                print(f"  轮次 {i+1}: 失败 - {e}")
                continue
        
        if times:
            avg_time = statistics.mean(times)
            avg_rtf = statistics.mean(rtfs) if rtfs else float('inf')
            
            results.append({
                'case': case_name,
                'text_length': len(text),
                'avg_time': avg_time,
                'avg_rtf': avg_rtf,
                'min_time': min(times),
                'max_time': max(times)
            })
            
            print(f"  平均: {avg_time:.2f}s, RTF: {avg_rtf:.3f}")
    
    return {
        'config': f"TRT={load_trt}, JIT={load_jit}, FP16={fp16}",
        'load_time': load_time,
        'results': results
    }

def main():
    parser = argparse.ArgumentParser(description='CosyVoice性能基准测试')
    parser.add_argument('--model_dir', type=str, 
                        default='pretrained_models/CosyVoice-300M-SFT',
                        help='模型路径')
    parser.add_argument('--test_all', action='store_true',
                        help='测试所有配置组合')
    parser.add_argument('--load_trt', action='store_true',
                        help='启用TensorRT')
    parser.add_argument('--load_jit', action='store_true', 
                        help='启用JIT')
    parser.add_argument('--fp16', action='store_true',
                        help='启用FP16')
    
    args = parser.parse_args()
    
    print("CosyVoice 性能基准测试")
    print(f"模型路径: {args.model_dir}")
    
    if not os.path.exists(args.model_dir):
        print(f"错误: 模型路径不存在: {args.model_dir}")
        return
    
    all_results = []
    
    if args.test_all:
        # 测试所有配置组合
        configs = [
            (False, False, False),  # 基础配置
            (False, True, False),   # 仅JIT
            (False, False, True),   # 仅FP16
            (False, True, True),    # JIT + FP16
            (True, False, False),   # 仅TRT
            (True, True, False),    # TRT + JIT
            (True, False, True),    # TRT + FP16
            (True, True, True),     # 全部启用
        ]
        
        for trt, jit, fp16 in configs:
            result = benchmark_config(args.model_dir, trt, jit, fp16)
            if result:
                all_results.append(result)
    else:
        # 测试指定配置
        result = benchmark_config(args.model_dir, args.load_trt, args.load_jit, args.fp16)
        if result:
            all_results.append(result)
    
    # 输出汇总结果
    print(f"\n{'='*80}")
    print("性能测试汇总")
    print(f"{'='*80}")
    
    for result in all_results:
        print(f"\n配置: {result['config']}")
        print(f"加载时间: {result['load_time']:.2f}s")
        
        for case_result in result['results']:
            print(f"  {case_result['case']}: {case_result['avg_time']:.2f}s, RTF: {case_result['avg_rtf']:.3f}")
    
    # RTF性能等级说明
    print(f"\n{'='*80}")
    print("RTF性能等级说明:")
    print("RTF < 0.3  : 🚀 非常快，实时性极佳")
    print("RTF 0.3-0.7: ⚡ 快速，适合实时应用") 
    print("RTF 0.7-1.0: ✅ 可接受，接近实时")
    print("RTF > 1.0  : 🐌 较慢，需要优化")

if __name__ == '__main__':
    main()
