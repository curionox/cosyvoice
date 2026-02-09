#!/usr/bin/env python3
"""
优化版 CosyVoice FastAPI 服务器启动脚本
自动应用最佳性能配置
"""
import sys
import os
import argparse
import subprocess
import time

def check_gpu_memory():
    """检查GPU内存"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存: {gpu_memory:.1f}GB")
            return gpu_memory
        else:
            print("未检测到CUDA GPU")
            return 0
    except:
        print("无法检测GPU信息")
        return 0

def recommend_config(gpu_memory_gb):
    """根据GPU内存推荐配置"""
    if gpu_memory_gb >= 8:
        return {
            'trt_concurrent': 2,
            'fp16': True,
            'load_trt': True,
            'load_jit': True,
            'note': '高性能配置 (8GB+ GPU)'
        }
    elif gpu_memory_gb >= 4:
        return {
            'trt_concurrent': 1,
            'fp16': True,
            'load_trt': True,
            'load_jit': True,
            'note': '标准配置 (4-8GB GPU)'
        }
    elif gpu_memory_gb >= 2:
        return {
            'trt_concurrent': 1,
            'fp16': True,
            'load_trt': False,
            'load_jit': True,
            'note': '轻量配置 (2-4GB GPU)'
        }
    else:
        return {
            'trt_concurrent': 1,
            'fp16': False,
            'load_trt': False,
            'load_jit': True,
            'note': 'CPU或低显存配置'
        }

def build_command(model_dir, port, config):
    """构建启动命令"""
    cmd = [
        sys.executable, 'fastapi_server.py',
        '--model_dir', model_dir,
        '--port', str(port),
        '--trt_concurrent', str(config['trt_concurrent'])
    ]
    
    if config['fp16']:
        cmd.append('--fp16')
    if config['load_trt']:
        cmd.append('--load_trt')
    if config['load_jit']:
        cmd.append('--load_jit')
    
    return cmd

def main():
    parser = argparse.ArgumentParser(description='优化版CosyVoice服务器启动器')
    parser.add_argument('--model_dir', type=str, 
                        default='pretrained_models/CosyVoice-300M-SFT',
                        help='模型路径')
    parser.add_argument('--port', type=int, default=9234,
                        help='服务端口')
    parser.add_argument('--auto_config', action='store_true', default=True,
                        help='自动配置优化参数 (默认启用)')
    parser.add_argument('--manual_config', action='store_true',
                        help='手动配置参数')
    parser.add_argument('--precompile', action='store_true',
                        help='启动前预编译TensorRT引擎')
    parser.add_argument('--benchmark', action='store_true',
                        help='启动前运行性能测试')
    
    # 手动配置选项
    parser.add_argument('--load_trt', action='store_true',
                        help='启用TensorRT')
    parser.add_argument('--load_jit', action='store_true',
                        help='启用JIT')
    parser.add_argument('--fp16', action='store_true',
                        help='启用FP16')
    parser.add_argument('--trt_concurrent', type=int, default=1,
                        help='TensorRT并发数')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CosyVoice 优化版服务器启动器")
    print("=" * 60)
    
    # 检查模型路径
    if not os.path.exists(args.model_dir):
        print(f"❌ 错误: 模型路径不存在: {args.model_dir}")
        return
    
    print(f"📁 模型路径: {args.model_dir}")
    print(f"🌐 服务端口: {args.port}")
    
    # 确定配置
    if args.manual_config:
        config = {
            'trt_concurrent': args.trt_concurrent,
            'fp16': args.fp16,
            'load_trt': args.load_trt,
            'load_jit': args.load_jit,
            'note': '手动配置'
        }
    else:
        # 自动配置
        print("\n🔍 检测硬件配置...")
        gpu_memory = check_gpu_memory()
        config = recommend_config(gpu_memory)
    
    print(f"\n⚙️  推荐配置: {config['note']}")
    print(f"   - TensorRT: {config['load_trt']}")
    print(f"   - JIT编译: {config['load_jit']}")
    print(f"   - FP16半精度: {config['fp16']}")
    print(f"   - TRT并发数: {config['trt_concurrent']}")
    
    # 预编译TensorRT引擎
    if args.precompile and config['load_trt']:
        print("\n🔥 预编译TensorRT引擎...")
        precompile_cmd = [
            sys.executable, 'precompile_trt.py',
            '--model_dir', args.model_dir,
            '--trt_concurrent', str(config['trt_concurrent'])
        ]
        if config['fp16']:
            precompile_cmd.append('--fp16')
        else:
            precompile_cmd.append('--no_fp16')
        
        try:
            subprocess.run(precompile_cmd, check=True)
            print("✅ TensorRT引擎预编译完成")
        except subprocess.CalledProcessError:
            print("⚠️  TensorRT预编译失败，继续启动服务器")
    
    # 性能测试
    if args.benchmark:
        print("\n📊 运行性能基准测试...")
        benchmark_cmd = [
            sys.executable, 'benchmark_performance.py',
            '--model_dir', args.model_dir
        ]
        if config['load_trt']:
            benchmark_cmd.append('--load_trt')
        if config['load_jit']:
            benchmark_cmd.append('--load_jit')
        if config['fp16']:
            benchmark_cmd.append('--fp16')
        
        try:
            subprocess.run(benchmark_cmd, check=True)
        except subprocess.CalledProcessError:
            print("⚠️  性能测试失败，继续启动服务器")
    
    # 构建启动命令
    cmd = build_command(args.model_dir, args.port, config)
    
    print(f"\n🚀 启动服务器...")
    print(f"命令: {' '.join(cmd)}")
    print(f"\n📖 服务文档: http://localhost:{args.port}/docs")
    print(f"🎯 API端点: http://localhost:{args.port}/")
    print(f"\n💡 优化提示:")
    print(f"   - 长文本会自动分割优化")
    print(f"   - 支持流式合成提升响应速度")
    print(f"   - 可调用 /warmup_trt 预热引擎")
    print(f"   - 查看 /performance_info 获取配置信息")
    
    print(f"\n{'='*60}")
    print("服务器启动中...")
    print("按 Ctrl+C 停止服务器")
    print(f"{'='*60}")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n👋 服务器已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")

if __name__ == '__main__':
    main()
