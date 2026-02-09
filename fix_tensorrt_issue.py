#!/usr/bin/env python3
"""
TensorRT 问题快速修复脚本
针对 "Unable to determine GPU memory usage" 等 TensorRT 初始化问题
"""
import sys
import os
import subprocess

def check_environment():
    """检查环境并提供修复建议"""
    print("🔍 检查 TensorRT 相关环境...")
    
    issues = []
    fixes = []
    
    # 检查 CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("CUDA 不可用")
            fixes.append("重新安装 CUDA 和 PyTorch GPU 版本")
        else:
            print(f"✅ CUDA 可用: {torch.version.cuda}")
    except ImportError:
        issues.append("PyTorch 未安装")
        fixes.append("安装 PyTorch GPU 版本")
    
    # 检查 ONNXRuntime
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' not in providers:
            issues.append("ONNXRuntime CUDA 提供程序不可用")
            fixes.append("安装 onnxruntime-gpu")
        else:
            print(f"✅ ONNXRuntime CUDA 提供程序可用")
    except ImportError:
        issues.append("ONNXRuntime 未安装")
        fixes.append("安装 onnxruntime-gpu")
    
    # 检查 TensorRT
    try:
        import tensorrt as trt
        print(f"✅ TensorRT 版本: {trt.__version__}")
    except ImportError:
        issues.append("TensorRT 未安装")
        fixes.append("安装 TensorRT")
    
    return issues, fixes

def apply_fixes():
    """应用自动修复"""
    print("\n🔧 开始自动修复...")
    
    fixes_applied = []
    
    # 修复 1: 更新 ONNXRuntime
    try:
        print("🔄 更新 ONNXRuntime...")
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'onnxruntime', '-y'], 
                      capture_output=True)
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'onnxruntime-gpu'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            fixes_applied.append("✅ ONNXRuntime GPU 版本安装成功")
        else:
            fixes_applied.append(f"❌ ONNXRuntime 安装失败: {result.stderr}")
    except Exception as e:
        fixes_applied.append(f"❌ ONNXRuntime 修复失败: {e}")
    
    # 修复 2: 检查 TensorRT 安装
    try:
        import tensorrt
    except ImportError:
        try:
            print("🔄 安装 TensorRT...")
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'tensorrt'], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                fixes_applied.append("✅ TensorRT 安装成功")
            else:
                fixes_applied.append(f"❌ TensorRT 安装失败: {result.stderr}")
        except Exception as e:
            fixes_applied.append(f"❌ TensorRT 安装失败: {e}")
    
    return fixes_applied

def generate_safe_configs():
    """生成安全的启动配置"""
    print("\n📋 生成安全启动配置...")
    
    configs = [
        {
            'name': '最安全配置 (仅 JIT)',
            'cmd': 'python fastapi_server.py --model_dir pretrained_models/CosyVoice-300M-SFT --port 9234 --load_jit',
            'desc': '避免 TensorRT 和 FP16，最稳定'
        },
        {
            'name': '中等配置 (JIT + FP16)',
            'cmd': 'python fastapi_server.py --model_dir pretrained_models/CosyVoice-300M-SFT --port 9234 --load_jit --fp16',
            'desc': '启用 FP16 加速，避免 TensorRT'
        },
        {
            'name': '基础配置 (无优化)',
            'cmd': 'python fastapi_server.py --model_dir pretrained_models/CosyVoice-300M-SFT --port 9234',
            'desc': '最基础配置，兼容性最好'
        }
    ]
    
    return configs

def create_startup_scripts():
    """创建启动脚本"""
    print("\n📝 创建启动脚本...")
    
    # 创建安全启动脚本
    safe_script = """@echo off
echo 启动 CosyVoice 安全配置...
python fastapi_server.py --model_dir pretrained_models/CosyVoice-300M-SFT --port 9234 --load_jit --fp16
pause
"""
    
    with open('start_safe.bat', 'w', encoding='utf-8') as f:
        f.write(safe_script)
    
    # 创建基础启动脚本
    basic_script = """@echo off
echo 启动 CosyVoice 基础配置...
python fastapi_server.py --model_dir pretrained_models/CosyVoice-300M-SFT --port 9234 --load_jit
pause
"""
    
    with open('start_basic.bat', 'w', encoding='utf-8') as f:
        f.write(basic_script)
    
    print("✅ 创建启动脚本:")
    print("   - start_safe.bat (JIT + FP16)")
    print("   - start_basic.bat (仅 JIT)")

def main():
    print("=" * 60)
    print("CosyVoice TensorRT 问题快速修复工具")
    print("=" * 60)
    
    # 检查环境
    issues, fixes = check_environment()
    
    if issues:
        print(f"\n❌ 发现 {len(issues)} 个问题:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        print(f"\n💡 建议修复:")
        for i, fix in enumerate(fixes, 1):
            print(f"   {i}. {fix}")
        
        # 询问是否自动修复
        try:
            choice = input("\n是否尝试自动修复? (y/n): ").lower().strip()
            if choice in ['y', 'yes', '是']:
                fixes_applied = apply_fixes()
                print("\n修复结果:")
                for fix in fixes_applied:
                    print(f"   {fix}")
        except KeyboardInterrupt:
            print("\n用户取消修复")
    else:
        print("\n✅ 环境检查通过")
    
    # 生成安全配置
    configs = generate_safe_configs()
    print("\n🚀 推荐启动配置:")
    for i, config in enumerate(configs, 1):
        print(f"\n{i}. {config['name']}")
        print(f"   命令: {config['cmd']}")
        print(f"   说明: {config['desc']}")
    
    # 创建启动脚本
    create_startup_scripts()
    
    print(f"\n{'='*60}")
    print("修复完成!")
    print("建议操作:")
    print("1. 重启终端/命令提示符")
    print("2. 运行: python diagnose_gpu.py (验证修复)")
    print("3. 使用: start_safe.bat 或 start_basic.bat 启动")
    print("4. 或运行: python start_fallback_server.py (自动选择配置)")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
