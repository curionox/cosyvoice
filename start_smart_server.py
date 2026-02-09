#!/usr/bin/env python3
"""
CosyVoice 智能启动脚本
自动处理TensorRT超时，智能回退到安全配置
"""
import sys
import os
import subprocess
import time
import signal
import threading

def run_with_timeout(cmd, timeout=600):  # 10分钟超时
    """运行命令并处理超时"""
    print(f"执行命令: {' '.join(cmd)}")
    print(f"超时设置: {timeout}秒")
    
    process = None
    result = {'returncode': None, 'timeout': False}
    
    def run_process():
        nonlocal process, result
        try:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                universal_newlines=True,
                bufsize=1
            )
            
            # 实时输出日志
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                    
            result['returncode'] = process.poll()
            
        except Exception as e:
            print(f"进程执行异常: {e}")
            result['returncode'] = -1
    
    # 启动进程线程
    thread = threading.Thread(target=run_process)
    thread.daemon = True
    thread.start()
    
    # 等待完成或超时
    thread.join(timeout)
    
    if thread.is_alive():
        result['timeout'] = True
        print(f"\n*** 进程超时 ({timeout}秒)，正在终止...")
        
        if process:
            try:
                # 尝试优雅终止
                process.terminate()
                time.sleep(5)
                
                if process.poll() is None:
                    # 强制终止
                    process.kill()
                    time.sleep(2)
                    
                print("进程已终止")
            except Exception as e:
                print(f"终止进程时出错: {e}")
        
        thread.join(5)  # 给线程5秒时间清理
    
    return result

def test_trt_configuration():
    """测试TensorRT配置是否可用"""
    print("=" * 60)
    print("测试TensorRT配置...")
    print("=" * 60)
    
    cmd = [
        sys.executable, 'fastapi_server.py',
        '--model_dir', 'pretrained_models/CosyVoice-300M-SFT',
        '--port', '9234',
        '--load_trt', '--load_jit', '--fp16'
    ]
    
    result = run_with_timeout(cmd, timeout=600)  # 10分钟超时
    
    if result['timeout']:
        print("❌ TensorRT配置测试超时")
        return False
    elif result['returncode'] == 0:
        print("✅ TensorRT配置测试成功")
        return True
    else:
        print(f"❌ TensorRT配置测试失败 (返回码: {result['returncode']})")
        return False

def start_fallback_server():
    """启动回退配置服务器"""
    print("=" * 60) 
    print("使用安全回退配置启动服务器...")
    print("配置: JIT + FP16 (无TensorRT)")
    print("=" * 60)
    
    cmd = [
        sys.executable, 'fastapi_server.py',
        '--model_dir', 'pretrained_models/CosyVoice-300M-SFT', 
        '--port', '9234',
        '--load_jit', '--fp16'
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"启动失败: {e}")

def main():
    print("=" * 60)
    print("CosyVoice 智能启动器")
    print("自动处理TensorRT超时问题") 
    print("=" * 60)
    
    # 检查模型路径
    model_dir = 'pretrained_models/CosyVoice-300M-SFT'
    if not os.path.exists(model_dir):
        print(f"错误: 模型路径不存在: {model_dir}")
        return
        
    print(f"模型路径: {model_dir}")
    print(f"服务端口: 9234")
    print()
    
    # 如果已存在TensorRT缓存文件，直接启动优化版本
    trt_cache_exists = False
    for root, dirs, files in os.walk(model_dir):
        if any(f.endswith('.trt') or f.endswith('.engine') for f in files):
            trt_cache_exists = True
            break
    
    if trt_cache_exists:
        print("发现TensorRT缓存文件，尝试直接启动优化版本...")
        cmd = [
            sys.executable, 'fastapi_server.py',
            '--model_dir', model_dir,
            '--port', '9234', 
            '--load_trt', '--load_jit', '--fp16'
        ]
        
        try:
            subprocess.run(cmd)
            return
        except KeyboardInterrupt:
            print("\n服务器已停止")
            return
        except Exception:
            print("优化版本启动失败，回退到安全配置...")
    
    # 测试TensorRT是否可用
    print("首次启动或TensorRT缓存不存在，进行配置测试...")
    
    # 使用更短的超时进行快速测试
    print("进行TensorRT兼容性快速测试 (3分钟超时)...")
    
    cmd = [
        sys.executable, 'fastapi_server.py',
        '--model_dir', model_dir,
        '--port', '9234',
        '--load_trt', '--load_jit', '--fp16'
    ]
    
    result = run_with_timeout(cmd, timeout=180)  # 3分钟快速测试
    
    if result['timeout']:
        print("\n" + "=" * 60)
        print("⚠️  TensorRT配置超时，自动回退到安全配置")
        print("=" * 60)
        start_fallback_server()
    elif result['returncode'] != 0:
        print(f"\n❌ TensorRT配置失败 (返回码: {result['returncode']})")
        print("自动回退到安全配置...")
        start_fallback_server()
    else:
        print("✅ TensorRT配置成功！")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序异常: {e}")
        print("\n建议:")
        print("1. 检查GPU驱动和CUDA环境")
        print("2. 尝试手动启动: python start_safe_rtx4060ti.bat")
        print("3. 检查显存是否足够")