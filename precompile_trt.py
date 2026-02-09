#!/usr/bin/env python3
"""
TensorRT 预编译脚本
用于预先生成 TensorRT 引擎文件，避免每次启动时的转换时间
"""
import sys
import os
import argparse
import time

# 添加必要的路径
sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2

def precompile_trt_engines(model_dir, fp16=True, trt_concurrent=1):
    """预编译TensorRT引擎"""
    print(f"开始预编译TensorRT引擎...")
    print(f"模型路径: {model_dir}")
    print(f"FP16: {fp16}, 并发数: {trt_concurrent}")
    
    start_time = time.time()
    
    # 加载模型
    try:
        cosyvoice = CosyVoice(model_dir, 
                             load_jit=True, 
                             load_trt=True, 
                             fp16=fp16, 
                             trt_concurrent=trt_concurrent)
        print("模型加载成功 (CosyVoice)")
    except Exception as e1:
        try:
            cosyvoice = CosyVoice2(model_dir, 
                                  load_jit=True, 
                                  load_trt=True, 
                                  fp16=fp16, 
                                  trt_concurrent=trt_concurrent)
            print("模型加载成功 (CosyVoice2)")
        except Exception as e2:
            print(f"模型加载失败:")
            print(f"  CosyVoice: {e1}")
            print(f"  CosyVoice2: {e2}")
            return False
    
    load_time = time.time() - start_time
    print(f"模型加载耗时: {load_time:.2f}s")
    
    # 获取可用音色
    try:
        voices = cosyvoice.list_available_spks()
        print(f"可用音色数量: {len(voices)}")
        sample_voice = voices[0] if voices else "中文女"
    except:
        sample_voice = "中文女"
        print("使用默认音色: 中文女")
    
    # 预热不同长度的文本
    test_texts = [
        "你好。",  # 短文本
        "这是一个中等长度的测试文本，用于预热TensorRT引擎。",  # 中等文本
        "这是一个比较长的测试文本，包含了多个句子。第一句话是介绍。第二句话是说明。第三句话是总结。通过这样的长文本可以更好地预热TensorRT引擎，确保后续推理的最佳性能。"  # 长文本
    ]
    
    print("\n开始预热TensorRT引擎...")
    warmup_start = time.time()
    
    for i, text in enumerate(test_texts):
        print(f"预热文本 {i+1}/{len(test_texts)} (长度: {len(text)})")
        try:
            # 执行推理来触发TensorRT编译
            output = list(cosyvoice.inference_sft(text, sample_voice, stream=False))
            print(f"  ✓ 预热成功")
        except Exception as e:
            print(f"  ✗ 预热失败: {e}")
    
    warmup_time = time.time() - warmup_start
    total_time = time.time() - start_time
    
    print(f"\n预编译完成!")
    print(f"预热耗时: {warmup_time:.2f}s")
    print(f"总耗时: {total_time:.2f}s")
    print(f"\n优化效果:")
    print(f"- TensorRT引擎已编译并缓存")
    print(f"- 后续启动将跳过编译过程")
    print(f"- 推理速度将显著提升")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='预编译TensorRT引擎')
    parser.add_argument('--model_dir', type=str, 
                        default='pretrained_models/CosyVoice-300M-SFT',
                        help='模型路径')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='启用FP16 (默认启用)')
    parser.add_argument('--no_fp16', action='store_true',
                        help='禁用FP16')
    parser.add_argument('--trt_concurrent', type=int, default=1,
                        help='TensorRT并发数')
    
    args = parser.parse_args()
    
    # 处理FP16参数
    fp16 = args.fp16 and not args.no_fp16
    
    print("=" * 60)
    print("CosyVoice TensorRT 预编译工具")
    print("=" * 60)
    
    if not os.path.exists(args.model_dir):
        print(f"错误: 模型路径不存在: {args.model_dir}")
        return
    
    success = precompile_trt_engines(args.model_dir, fp16, args.trt_concurrent)
    
    if success:
        print("\n🎉 预编译成功!")
        print("\n下次启动服务器时使用:")
        print(f"python fastapi_server.py --model_dir {args.model_dir} --load_trt --load_jit --fp16 --port 9234")
    else:
        print("\n❌ 预编译失败!")

if __name__ == '__main__':
    main()
