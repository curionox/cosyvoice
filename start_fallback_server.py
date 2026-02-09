#!/usr/bin/env python3
"""
CosyVoice 备用启动脚本
提供多种优化级别，自动降级到可用配置
"""
import sys
import os
import argparse
import subprocess
import time

def test_config(model_dir, config_name, **kwargs):
    """测试特定配置是否可用"""
    print(f"测试 {config_name} 配置...")
    
    # 添加路径
    sys.path.append('third_party/Matcha-TTS')
    
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
        
        # 尝试加载模型
        try:
            cosyvoice = CosyVoice(model_dir, **kwargs)
            print(f"成功: {config_name} 配置测试成功")
            del cosyvoice
            return True
        except Exception as e1:
            try:
                cosyvoice = CosyVoice2(model_dir, **kwargs)
                print(f"成功: {config_name} 配置测试成功 (CosyVoice2)")
                del cosyvoice
                return True
            except Exception as e2:
                print(f"失败: {config_name} 配置测试失败: {e1}")
                return False
                
    except ImportError as e:
        print(f"导入失败: {e}")
        return False

def find_working_config(model_dir):
    """找到可用的最佳配置"""
    print("自动检测可用配置...")
    
    # 配置列表，按优化程度排序
    configs = [
        {
            'name': '完整优化 (TRT+JIT+FP16)',
            'args': {'load_trt': True, 'load_jit': True, 'fp16': True},
            'cmd_args': ['--load_trt', '--load_jit', '--fp16']
        },
        {
            'name': 'TRT+FP16',
            'args': {'load_trt': True, 'fp16': True},
            'cmd_args': ['--load_trt', '--fp16']
        },
        {
            'name': 'JIT+FP16',
            'args': {'load_jit': True, 'fp16': True},
            'cmd_args': ['--load_jit', '--fp16']
        },
        {
            'name': '仅JIT',
            'args': {'load_jit': True},
            'cmd_args': ['--load_jit']
        },
        {
            'name': '仅FP16',
            'args': {'fp16': True},
            'cmd_args': ['--fp16']
        },
        {
            'name': '基础配置',
            'args': {},
            'cmd_args': []
        }
    ]
    
    for config in configs:
        if test_config(model_dir, config['name'], **config['args']):
            return config
    
    return None

def main():
    parser = argparse.ArgumentParser(description='CosyVoice 备用启动器')
    parser.add_argument('--model_dir', type=str, 
                        default='pretrained_models/CosyVoice-300M-SFT',
                        help='模型路径')
    parser.add_argument('--port', type=int, default=9234,
                        help='服务端口')
    parser.add_argument('--force_config', type=str,
                        choices=['full', 'trt_fp16', 'jit_fp16', 'jit', 'fp16', 'basic'],
                        help='强制使用指定配置')
    parser.add_argument('--skip_test', action='store_true',
                        help='跳过配置测试，直接启动')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CosyVoice 备用启动器")
    print("=" * 60)
    
    # 检查模型路径
    if not os.path.exists(args.model_dir):
        print(f"错误: 模型路径不存在: {args.model_dir}")
        return
    
    print(f"模型路径: {args.model_dir}")
    print(f"服务端口: {args.port}")
    
    # 强制配置映射
    force_configs = {
        'full': {
            'name': '完整优化 (强制)',
            'cmd_args': ['--load_trt', '--load_jit', '--fp16']
        },
        'trt_fp16': {
            'name': 'TRT+FP16 (强制)',
            'cmd_args': ['--load_trt', '--fp16']
        },
        'jit_fp16': {
            'name': 'JIT+FP16 (强制)',
            'cmd_args': ['--load_jit', '--fp16']
        },
        'jit': {
            'name': '仅JIT (强制)',
            'cmd_args': ['--load_jit']
        },
        'fp16': {
            'name': '仅FP16 (强制)',
            'cmd_args': ['--fp16']
        },
        'basic': {
            'name': '基础配置 (强制)',
            'cmd_args': []
        }
    }
    
    if args.force_config:
        config = force_configs[args.force_config]
        print(f"\n强制使用配置: {config['name']}")
    elif args.skip_test:
        # 默认使用中等配置
        config = {
            'name': 'JIT+FP16 (默认)',
            'cmd_args': ['--load_jit', '--fp16']
        }
        print(f"\n使用默认配置: {config['name']}")
    else:
        # 自动检测最佳配置
        config = find_working_config(args.model_dir)
        if not config:
            print("没有找到可用的配置")
            return
        print(f"\n选择配置: {config['name']}")
    
    # 构建启动命令
    cmd = [
        sys.executable, 'fastapi_server.py',
        '--model_dir', args.model_dir,
        '--port', str(args.port)
    ]
    cmd.extend(config['cmd_args'])
    
    print(f"\n启动服务器...")
    print(f"命令: {' '.join(cmd)}")
    
    # 显示服务信息
    print(f"\n服务文档: http://localhost:{args.port}/docs")
    print(f"API端点: http://localhost:{args.port}/")
    print(f"性能信息: http://localhost:{args.port}/performance_info")
    
    # 根据配置给出优化建议
    if 'load_trt' not in config['cmd_args']:
        print(f"\n优化建议:")
        print(f"   - 如果遇到性能问题，可以运行诊断脚本:")
        print(f"     python diagnose_gpu.py")
        print(f"   - 尝试安装 TensorRT 以获得更好性能")
    
    print(f"\n{'='*60}")
    print("服务器启动中...")
    print("按 Ctrl+C 停止服务器")
    print(f"{'='*60}")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n服务器已停止")
    except Exception as e:
        print(f"\n启动失败: {e}")
        print("\n建议:")
        print("1. 运行诊断脚本: python diagnose_gpu.py")
        print("2. 尝试更基础的配置: python start_fallback_server.py --force_config basic")
        print("3. 检查模型文件是否完整")

if __name__ == '__main__':
    main()
