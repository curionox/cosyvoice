#!/usr/bin/env python3
"""
RTX 4060 Ti 16GB 专用优化启动脚本
基于诊断结果的定制化配置
"""
import sys
import os
import subprocess
import time

def test_tensorrt_compatibility():
    """测试 TensorRT 兼容性"""
    print("🔄 测试 TensorRT 兼容性...")
    
    # 添加路径
    sys.path.append('third_party/Matcha-TTS')
    
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
        
        model_dir = 'pretrained_models/CosyVoice-300M-SFT'
        if not os.path.exists(model_dir):
            print(f"❌ 模型路径不存在: {model_dir}")
            return False
        
        # 测试 TensorRT 加载（短时间测试）
        print("🔄 测试 TensorRT 模型加载（30秒超时）...")
        
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("TensorRT 加载超时")
        
        # Windows 不支持 signal.alarm，使用其他方法
        try:
            start_time = time.time()
            cosyvoice = CosyVoice(model_dir, load_trt=True, fp16=True, load_jit=True)
            load_time = time.time() - start_time
            print(f"✅ TensorRT 模型加载成功，耗时: {load_time:.1f}s")
            del cosyvoice
            return True
        except Exception as e:
            print(f"❌ TensorRT 模型加载失败: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ 模型导入失败: {e}")
        return False

def get_optimal_config():
    """获取 RTX 4060 Ti 16GB 的最优配置"""
    
    # RTX 4060 Ti 16GB 推荐配置
    configs = [
        {
            'name': 'RTX 4060 Ti 高性能配置',
            'args': ['--load_trt', '--load_jit', '--fp16', '--trt_concurrent', '2'],
            'desc': '启用所有优化，并发数2（16GB显存充足）'
        },
        {
            'name': 'RTX 4060 Ti 标准配置',
            'args': ['--load_trt', '--load_jit', '--fp16'],
            'desc': '启用所有优化，单并发（推荐）'
        },
        {
            'name': 'RTX 4060 Ti 安全配置',
            'args': ['--load_jit', '--fp16'],
            'desc': '避免 TensorRT，使用 JIT + FP16'
        },
        {
            'name': 'RTX 4060 Ti 基础配置',
            'args': ['--load_jit'],
            'desc': '仅使用 JIT 优化'
        }
    ]
    
    return configs

def main():
    print("=" * 60)
    print("RTX 4060 Ti 16GB 专用优化启动器")
    print("=" * 60)
    
    model_dir = 'pretrained_models/CosyVoice-300M-SFT'
    port = 9234
    
    # 检查模型路径
    if not os.path.exists(model_dir):
        print(f"❌ 错误: 模型路径不存在: {model_dir}")
        return
    
    print(f"📁 模型路径: {model_dir}")
    print(f"🌐 服务端口: {port}")
    print(f"🎮 GPU: RTX 4060 Ti 16GB")
    
    # 获取配置选项
    configs = get_optimal_config()
    
    print(f"\n⚙️  可用配置:")
    for i, config in enumerate(configs, 1):
        print(f"{i}. {config['name']}")
        print(f"   {config['desc']}")
    
    # 用户选择配置
    try:
        print(f"\n请选择配置 (1-{len(configs)}) 或按 Enter 使用推荐配置 [2]: ", end='')
        choice = input().strip()
        
        if not choice:
            choice = '2'  # 默认使用标准配置
        
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(configs):
            selected_config = configs[choice_idx]
        else:
            print("无效选择，使用标准配置")
            selected_config = configs[1]
            
    except (ValueError, KeyboardInterrupt):
        print("使用标准配置")
        selected_config = configs[1]
    
    print(f"\n✅ 选择配置: {selected_config['name']}")
    
    # 构建启动命令
    cmd = [
        sys.executable, 'fastapi_server.py',
        '--model_dir', model_dir,
        '--port', str(port)
    ]
    cmd.extend(selected_config['args'])
    
    print(f"\n🚀 启动命令: {' '.join(cmd)}")
    
    # 显示服务信息
    print(f"\n📖 服务文档: http://localhost:{port}/docs")
    print(f"🎯 API端点: http://localhost:{port}/")
    print(f"🔧 性能信息: http://localhost:{port}/performance_info")
    
    # RTX 4060 Ti 特定优化提示
    print(f"\n💡 RTX 4060 Ti 16GB 优化提示:")
    print(f"   - 16GB 显存充足，可以使用高并发配置")
    print(f"   - 支持 FP16 和 TensorRT 加速")
    print(f"   - 如果 TensorRT 有问题，会自动降级到 JIT+FP16")
    print(f"   - 预期 RTF < 0.5 (非常快)")
    
    # 特殊处理：如果选择了 TensorRT 配置，先测试兼容性
    if '--load_trt' in selected_config['args']:
        print(f"\n🔄 检测到 TensorRT 配置，进行兼容性测试...")
        print(f"⚠️  如果卡住超过 30 秒，请按 Ctrl+C 取消并选择安全配置")
    
    print(f"\n{'='*60}")
    print("服务器启动中...")
    print("按 Ctrl+C 停止服务器")
    print(f"{'='*60}")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n👋 服务器已停止")
        
        # 如果是 TensorRT 配置失败，建议降级
        if '--load_trt' in selected_config['args']:
            print("\n💡 如果 TensorRT 启动有问题，建议使用安全配置:")
            print("python start_rtx4060ti_optimized.py")
            print("然后选择配置 3 (安全配置)")
            
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        
        if '--load_trt' in selected_config['args']:
            print("\n🔧 TensorRT 启动失败，尝试安全配置:")
            print("python fastapi_server.py --model_dir pretrained_models/CosyVoice-300M-SFT --port 9234 --load_jit --fp16")

if __name__ == '__main__':
    main()
