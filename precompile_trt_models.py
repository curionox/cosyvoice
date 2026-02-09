#!/usr/bin/env python3
"""
预编译TensorRT模型脚本
单独进行TensorRT转换，避免启动时卡住
"""
import os
import sys
import time
import logging
import threading
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
    # 添加第三方库路径
    third_party_path = Path(__file__).parent / 'third_party' / 'Matcha-TTS'
    if third_party_path.exists():
        sys.path.insert(0, str(third_party_path))
        logging.info(f"添加路径: {third_party_path}")

def check_gpu_memory():
    """检查GPU显存"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_free = torch.cuda.memory_reserved(0) / (1024**3)
            logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logging.info(f"GPU总显存: {gpu_memory:.1f}GB")
            logging.info(f"GPU可用显存: {gpu_memory - gpu_free:.1f}GB") 
            
            if gpu_memory < 8.0:
                logging.warning("显存不足8GB，TensorRT转换可能失败")
                return False
            return True
        else:
            logging.error("CUDA不可用")
            return False
    except Exception as e:
        logging.error(f"GPU检查失败: {e}")
        return False

def precompile_model(model_dir, use_fp16=True):
    """预编译模型的TensorRT组件"""
    logging.info("=" * 60)
    logging.info("开始预编译TensorRT模型")
    logging.info("=" * 60)
    
    if not os.path.exists(model_dir):
        logging.error(f"模型路径不存在: {model_dir}")
        return False
    
    try:
        # 导入CosyVoice
        from cosyvoice.cli.cosyvoice import CosyVoice
        from cosyvoice.utils.file_utils import convert_onnx_to_trt
        
        logging.info(f"模型路径: {model_dir}")
        logging.info(f"使用FP16: {use_fp16}")
        
        # 初始化模型但不立即转换TensorRT
        logging.info("加载基础模型...")
        cosyvoice = CosyVoice(
            model_dir, 
            load_jit=False,
            load_trt=False,  # 先不加载TRT
            fp16=use_fp16
        )
        
        logging.info("基础模型加载完成")
        
        # 查找ONNX模型文件
        onnx_files = []
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith('.onnx'):
                    onnx_files.append(os.path.join(root, file))
        
        if not onnx_files:
            logging.warning("未找到ONNX文件，尝试加载TensorRT模型...")
            # 尝试直接加载TRT模型
            cosyvoice_trt = CosyVoice(
                model_dir,
                load_trt=True,
                load_jit=True,
                fp16=use_fp16
            )
            logging.info("TensorRT模型加载成功")
            return True
        
        logging.info(f"找到ONNX文件: {len(onnx_files)}个")
        for onnx_file in onnx_files:
            logging.info(f"  - {onnx_file}")
        
        # 预编译每个ONNX文件
        success_count = 0
        for i, onnx_file in enumerate(onnx_files):
            logging.info(f"\n[{i+1}/{len(onnx_files)}] 转换: {os.path.basename(onnx_file)}")
            
            # 生成TRT文件名
            trt_file = onnx_file.replace('.onnx', '.trt')
            if os.path.exists(trt_file):
                logging.info(f"TRT文件已存在: {trt_file}")
                success_count += 1
                continue
            
            try:
                # 模拟TRT转换参数（需要根据实际模型调整）
                trt_kwargs = {
                    'input_names': ['input'],  # 需要根据实际模型调整
                    'min_shape': [(1, 80, 1)],  # 需要根据实际模型调整
                    'opt_shape': [(1, 80, 500)],  # 需要根据实际模型调整
                    'max_shape': [(1, 80, 2000)]  # 需要根据实际模型调整
                }
                
                # 执行转换
                start_time = time.time()
                convert_onnx_to_trt(trt_file, trt_kwargs, onnx_file, use_fp16)
                elapsed = time.time() - start_time
                
                if os.path.exists(trt_file):
                    file_size = os.path.getsize(trt_file) / (1024*1024)
                    logging.info(f"转换成功: {trt_file} ({file_size:.1f}MB, 耗时{elapsed:.1f}秒)")
                    success_count += 1
                else:
                    logging.error(f"转换失败: 未生成TRT文件")
                    
            except Exception as e:
                logging.error(f"转换ONNX文件失败 {onnx_file}: {e}")
                continue
        
        logging.info(f"\n转换完成: {success_count}/{len(onnx_files)} 个文件成功")
        
        if success_count > 0:
            # 测试加载转换后的模型
            logging.info("\n测试加载TensorRT优化模型...")
            try:
                cosyvoice_final = CosyVoice(
                    model_dir,
                    load_trt=True,
                    load_jit=True, 
                    fp16=use_fp16
                )
                logging.info("✅ TensorRT优化模型加载成功!")
                return True
            except Exception as e:
                logging.error(f"TensorRT模型加载测试失败: {e}")
                return False
        else:
            logging.error("所有ONNX文件转换都失败了")
            return False
            
    except ImportError as e:
        logging.error(f"模块导入失败: {e}")
        logging.error("请确保在正确的conda环境中运行此脚本")
        return False
    except Exception as e:
        logging.error(f"预编译失败: {e}")
        return False

def main():
    print("=" * 60)
    print("CosyVoice TensorRT 预编译工具")
    print("=" * 60)
    
    # 设置环境
    setup_environment()
    
    # 检查GPU
    if not check_gpu_memory():
        print("GPU环境检查失败，继续尝试...")
    
    # 模型路径
    model_dir = "pretrained_models/CosyVoice-300M-SFT"
    
    if not os.path.exists(model_dir):
        print(f"错误: 模型路径不存在: {model_dir}")
        print("请确保已下载CosyVoice模型")
        return
    
    print(f"模型路径: {model_dir}")
    
    # 选择配置
    print("\n选择转换配置:")
    print("1. FP16 (推荐，更快)")
    print("2. FP32 (更稳定)")
    
    try:
        choice = input("请选择 (1/2, 默认1): ").strip()
        use_fp16 = choice != '2'
    except EOFError:
        print("使用默认配置: FP16")
        use_fp16 = True
    
    print(f"\n配置: {'FP16' if use_fp16 else 'FP32'}")
    print("开始预编译，这可能需要几分钟时间...")
    print("日志将保存到 trt_conversion.log")
    
    # 执行预编译
    success = precompile_model(model_dir, use_fp16)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ TensorRT预编译完成!")
        print("现在可以正常启动服务器:")
        print("python fastapi_server.py --model_dir pretrained_models/CosyVoice-300M-SFT --port 9234 --load_trt --load_jit --fp16")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ TensorRT预编译失败")
        print("建议:")
        print("1. 检查日志文件: trt_conversion.log")
        print("2. 尝试使用安全配置启动:")
        print("   python fastapi_server.py --model_dir pretrained_models/CosyVoice-300M-SFT --port 9234 --load_jit --fp16")
        print("3. 检查显存是否足够 (建议8GB+)")
        print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断了预编译过程")
    except Exception as e:
        print(f"\n程序异常: {e}")
        logging.error(f"程序异常: {e}", exc_info=True)