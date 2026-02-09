#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
torchaudio DLL 依赖诊断脚本
用于诊断 Windows 上 torchaudio 的 DLL 依赖问题
"""

import os
import sys
import ctypes
import subprocess
from pathlib import Path

def check_file_exists():
    """检查 libtorchaudio.pyd 文件是否存在"""
    pyd_path = r"C:\Users\yehh\.conda\envs\cosyvoice\Lib\site-packages\torchaudio\lib\libtorchaudio.pyd"
    
    print("=== 文件存在性检查 ===")
    if os.path.exists(pyd_path):
        print(f"✓ 文件存在: {pyd_path}")
        file_size = os.path.getsize(pyd_path)
        print(f"  文件大小: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        return True
    else:
        print(f"✗ 文件不存在: {pyd_path}")
        return False

def check_torchaudio_lib_directory():
    """检查 torchaudio/lib 目录内容"""
    lib_dir = r"C:\Users\yehh\.conda\envs\cosyvoice\Lib\site-packages\torchaudio\lib"
    
    print("\n=== torchaudio/lib 目录内容 ===")
    if os.path.exists(lib_dir):
        files = os.listdir(lib_dir)
        print(f"目录: {lib_dir}")
        print(f"文件数量: {len(files)}")
        for file in sorted(files):
            file_path = os.path.join(lib_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"  - {file} ({size:,} bytes)")
            else:
                print(f"  - {file} (目录)")
    else:
        print(f"✗ 目录不存在: {lib_dir}")

def check_torch_lib_directory():
    """检查 torch/lib 目录内容"""
    lib_dir = r"C:\Users\yehh\.conda\envs\cosyvoice\Lib\site-packages\torch\lib"
    
    print("\n=== torch/lib 目录内容 ===")
    if os.path.exists(lib_dir):
        files = os.listdir(lib_dir)
        print(f"目录: {lib_dir}")
        print(f"文件数量: {len(files)}")
        # 只显示 DLL 文件
        dll_files = [f for f in files if f.endswith('.dll')]
        for file in sorted(dll_files):
            file_path = os.path.join(lib_dir, file)
            size = os.path.getsize(file_path)
            print(f"  - {file} ({size:,} bytes)")
    else:
        print(f"✗ 目录不存在: {lib_dir}")

def test_direct_load():
    """尝试直接加载 libtorchaudio.pyd"""
    pyd_path = r"C:\Users\yehh\.conda\envs\cosyvoice\Lib\site-packages\torchaudio\lib\libtorchaudio.pyd"
    
    print("\n=== 直接加载测试 ===")
    try:
        # 尝试直接加载
        lib = ctypes.CDLL(pyd_path)
        print("✓ libtorchaudio.pyd 直接加载成功")
        return True
    except OSError as e:
        print(f"✗ 直接加载失败: {e}")
        return False

def test_python_import():
    """测试 Python 导入"""
    print("\n=== Python 导入测试 ===")
    
    # 测试基本导入
    try:
        import torch
        print("✓ torch 导入成功")
        print(f"  torch 版本: {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"✗ torch 导入失败: {e}")
        return False
    
    # 测试 torchaudio 导入
    try:
        import torchaudio
        print("✓ torchaudio 导入成功")
        print(f"  torchaudio 版本: {torchaudio.__version__}")
        return True
    except ImportError as e:
        print(f"✗ torchaudio 导入失败: {e}")
        return False

def check_system_dlls():
    """检查系统 DLL"""
    print("\n=== 系统 DLL 检查 ===")
    
    # 常见的依赖 DLL
    common_dlls = [
        "msvcr120.dll",
        "msvcp120.dll", 
        "msvcr140.dll",
        "msvcp140.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "libiomp5md.dll",
        "mkl_core.dll",
        "mkl_intel_thread.dll",
        "mkl_rt.dll"
    ]
    
    for dll in common_dlls:
        try:
            ctypes.windll.kernel32.LoadLibraryW(dll)
            print(f"✓ {dll} 可用")
        except OSError:
            print(f"✗ {dll} 不可用")

def check_conda_env_paths():
    """检查 conda 环境路径"""
    print("\n=== Conda 环境路径检查 ===")
    
    conda_env = r"C:\Users\yehh\.conda\envs\cosyvoice"
    paths_to_check = [
        os.path.join(conda_env, "Library", "bin"),
        os.path.join(conda_env, "Scripts"),
        os.path.join(conda_env, "DLLs"),
        os.path.join(conda_env, "Lib", "site-packages", "torch", "lib"),
        os.path.join(conda_env, "Lib", "site-packages", "torchaudio", "lib"),
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            dll_count = len([f for f in os.listdir(path) if f.endswith('.dll')])
            print(f"✓ {path} (包含 {dll_count} 个 DLL)")
        else:
            print(f"✗ {path} (不存在)")

def main():
    """主函数"""
    print("torchaudio DLL 依赖诊断工具")
    print("=" * 50)
    
    # 1. 检查文件存在性
    if not check_file_exists():
        return
    
    # 2. 检查目录内容
    check_torchaudio_lib_directory()
    check_torch_lib_directory()
    
    # 3. 检查 conda 环境路径
    check_conda_env_paths()
    
    # 4. 检查系统 DLL
    check_system_dlls()
    
    # 5. 测试直接加载
    direct_load_ok = test_direct_load()
    
    # 6. 测试 Python 导入
    python_import_ok = test_python_import()
    
    print("\n" + "=" * 50)
    print("诊断总结:")
    print(f"- 直接加载: {'成功' if direct_load_ok else '失败'}")
    print(f"- Python 导入: {'成功' if python_import_ok else '失败'}")
    
    if not direct_load_ok or not python_import_ok:
        print("\n建议的解决方案:")
        print("1. 运行修复脚本: python fix_torchaudio_deps.py")
        print("2. 或者重新安装 torchaudio")
        print("3. 检查是否缺少 Visual C++ Redistributable")

if __name__ == "__main__":
    main()
