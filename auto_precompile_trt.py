#!/usr/bin/env python3
"""
自动预编译TensorRT模型 - 无交互版本
"""
import os
import sys
import time
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trt_conversion.log', encoding='utf-8')
    ]
)

def setup_environment():
    """设置环境"""
    third_party_path = Path(__file__).parent / 'third_party' / 'Matcha-TTS'
    if third_party_path.exists():
        sys.path.insert(0, str(third_party_path))
        logging.info(f"添加路径: {third_party_path}")

def precompile_model_simple(model_dir):
    """简单的预编译 - 直接尝试加载TRT模型来触发转换"""
    logging.info("=" * 60)
    logging.info("自动预编译TensorRT模型")
    logging.info("=" * 60)
    
    try:
        # 设置环境
        setup_environment()
        
        from cosyvoice.cli.cosyvoice import CosyVoice
        
        logging.info(f"模型路径: {model_dir}")
        logging.info("配置: TRT + JIT + FP16")
        
        # 直接尝试加载TRT模型，这会触发自动转换
        start_time = time.time()
        logging.info("开始加载模型并转换TensorRT...")
        
        cosyvoice = CosyVoice(
            model_dir,
            load_trt=True,
            load_jit=True, 
            fp16=True
        )
        
        elapsed = time.time() - start_time
        logging.info(f"✅ 模型加载和TensorRT转换成功! 耗时: {elapsed:.1f}秒")
        
        # 清理内存
        del cosyvoice
        
        return True
        
    except Exception as e:
        logging.error(f"预编译失败: {e}")
        return False

def main():
    print("=" * 60)
    print("CosyVoice TensorRT 自动预编译")
    print("=" * 60)
    
    model_dir = "pretrained_models/CosyVoice-300M-SFT"
    
    if not os.path.exists(model_dir):
        print(f"错误: 模型路径不存在: {model_dir}")
        return
    
    print(f"模型路径: {model_dir}")
    print("配置: TensorRT + JIT + FP16")
    print("开始自动预编译...")
    print("(此过程可能需要5-10分钟，请耐心等待)")
    print()
    
    success = precompile_model_simple(model_dir)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ TensorRT预编译成功!")
        print("现在可以快速启动服务器:")
        print("python fastapi_server.py --model_dir pretrained_models/CosyVoice-300M-SFT --port 9234 --load_trt --load_jit --fp16")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ TensorRT预编译失败")
        print("请检查日志: trt_conversion.log")
        print("建议使用安全配置启动服务器")
        print("=" * 60)

if __name__ == "__main__":
    main()