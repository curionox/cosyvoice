#!/usr/bin/env python3
"""
GPU 和 CUDA 环境诊断脚本
用于排查 TensorRT 和 CUDA 相关问题
"""
import sys
import os

def check_cuda():
    """检查 CUDA 环境"""
    print("🔍 检查 CUDA 环境...")
    
    try:
        import torch
        print(f"✅ PyTorch 版本: {torch.__version__}")
        print(f"✅ CUDA 可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA 版本: {torch.version.cuda}")
            print(f"✅ GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
                
                # 测试 GPU 内存分配
                try:
                    test_tensor = torch.randn(100, 100).cuda(i)
                    print(f"   ✅ GPU {i} 内存分配测试成功")
                    del test_tensor
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"   ❌ GPU {i} 内存分配测试失败: {e}")
        else:
            print("❌ CUDA 不可用")
            return False
            
    except ImportError:
        print("❌ PyTorch 未安装")
        return False
    
    return True

def check_onnxruntime():
    """检查 ONNXRuntime"""
    print("\n🔍 检查 ONNXRuntime...")
    
    try:
        import onnxruntime as ort
        print(f"✅ ONNXRuntime 版本: {ort.__version__}")
        
        providers = ort.get_available_providers()
        print(f"✅ 可用提供程序: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDA 执行提供程序可用")
            
            # 测试 CUDA 提供程序
            try:
                session = ort.InferenceSession(
                    b'<onnx model placeholder>', 
                    providers=['CUDAExecutionProvider']
                )
                print("✅ CUDA 提供程序测试成功")
            except Exception as e:
                print(f"⚠️  CUDA 提供程序测试失败: {e}")
            
            return True
        else:
            print("❌ CUDA 执行提供程序不可用")
            print("💡 建议安装 onnxruntime-gpu:")
            print("   pip uninstall onnxruntime")
            print("   pip install onnxruntime-gpu")
            return False
            
    except ImportError:
        print("❌ ONNXRuntime 未安装")
        return False

def check_tensorrt():
    """检查 TensorRT"""
    print("\n🔍 检查 TensorRT...")
    
    try:
        import tensorrt as trt
        print(f"✅ TensorRT 版本: {trt.__version__}")
        
        # 检查 TensorRT 构建器
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            print("✅ TensorRT 构建器创建成功")
            
            # 检查 GPU 设备
            if builder.max_DLA_batch_size >= 0:
                print("✅ DLA 设备可用")
            
            print(f"✅ 最大批次大小: {builder.max_batch_size}")
            print(f"✅ 最大工作空间大小: {builder.max_workspace_size}")
            
        except Exception as e:
            print(f"❌ TensorRT 构建器测试失败: {e}")
            return False
            
        return True
        
    except ImportError:
        print("❌ TensorRT 未安装")
        print("💡 建议安装 TensorRT:")
        print("   pip install tensorrt")
        return False

def check_system_info():
    """检查系统信息"""
    print("\n🔍 检查系统信息...")
    
    import platform
    print(f"✅ 操作系统: {platform.system()} {platform.release()}")
    print(f"✅ Python 版本: {platform.python_version()}")
    
    # 检查 NVIDIA 驱动
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    print(f"✅ NVIDIA 驱动: {line.split('Driver Version: ')[1].split()[0]}")
                    break
        else:
            print("❌ nvidia-smi 命令失败")
    except Exception as e:
        print(f"❌ 无法检查 NVIDIA 驱动: {e}")

def test_model_loading():
    """测试模型加载"""
    print("\n🔍 测试 CosyVoice 模型加载...")
    
    # 添加路径
    sys.path.append('third_party/Matcha-TTS')
    
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
        
        model_dir = 'pretrained_models/CosyVoice-300M-SFT'
        if not os.path.exists(model_dir):
            print(f"❌ 模型路径不存在: {model_dir}")
            return False
        
        print("🔄 测试基础模型加载...")
        try:
            cosyvoice = CosyVoice(model_dir)
            print("✅ 基础模型加载成功")
            del cosyvoice
        except Exception as e:
            print(f"❌ 基础模型加载失败: {e}")
        
        print("🔄 测试 JIT 模型加载...")
        try:
            cosyvoice = CosyVoice(model_dir, load_jit=True)
            print("✅ JIT 模型加载成功")
            del cosyvoice
        except Exception as e:
            print(f"❌ JIT 模型加载失败: {e}")
        
        print("🔄 测试 FP16 模型加载...")
        try:
            cosyvoice = CosyVoice(model_dir, fp16=True)
            print("✅ FP16 模型加载成功")
            del cosyvoice
        except Exception as e:
            print(f"❌ FP16 模型加载失败: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ CosyVoice 导入失败: {e}")
        return False

def main():
    print("=" * 60)
    print("CosyVoice GPU 环境诊断工具")
    print("=" * 60)
    
    results = {
        'cuda': check_cuda(),
        'onnxruntime': check_onnxruntime(),
        'tensorrt': check_tensorrt(),
        'model': False
    }
    
    check_system_info()
    
    if results['cuda']:
        results['model'] = test_model_loading()
    
    print("\n" + "=" * 60)
    print("诊断结果汇总")
    print("=" * 60)
    
    print(f"CUDA 环境: {'✅ 正常' if results['cuda'] else '❌ 异常'}")
    print(f"ONNXRuntime: {'✅ 正常' if results['onnxruntime'] else '❌ 异常'}")
    print(f"TensorRT: {'✅ 正常' if results['tensorrt'] else '❌ 异常'}")
    print(f"模型加载: {'✅ 正常' if results['model'] else '❌ 异常'}")
    
    print("\n💡 建议的启动配置:")
    
    if all(results.values()):
        print("🚀 推荐使用完整优化配置:")
        print("python fastapi_server.py --model_dir pretrained_models/CosyVoice-300M-SFT --port 9234 --load_trt --load_jit --fp16")
    elif results['cuda'] and results['onnxruntime']:
        print("⚡ 推荐使用中等优化配置:")
        print("python fastapi_server.py --model_dir pretrained_models/CosyVoice-300M-SFT --port 9234 --load_jit --fp16")
    elif results['cuda']:
        print("✅ 推荐使用基础优化配置:")
        print("python fastapi_server.py --model_dir pretrained_models/CosyVoice-300M-SFT --port 9234 --load_jit")
    else:
        print("🐌 推荐使用 CPU 配置:")
        print("python fastapi_server.py --model_dir pretrained_models/CosyVoice-300M-SFT --port 9234")
    
    print("\n🔧 故障排除建议:")
    if not results['cuda']:
        print("- 检查 CUDA 安装和 GPU 驱动")
        print("- 重新安装 PyTorch GPU 版本")
    if not results['onnxruntime']:
        print("- 安装 onnxruntime-gpu 替换 onnxruntime")
    if not results['tensorrt']:
        print("- 安装 TensorRT 或使用非 TRT 配置")
        print("- 检查 TensorRT 与 CUDA 版本兼容性")

if __name__ == '__main__':
    main()
