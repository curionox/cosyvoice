#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CosyVoice 专业流式音频播放器
使用sounddevice实现真正的实时流式播放
"""

import requests
import sounddevice as sd
import numpy as np
import threading
import queue
import time
import io
import wave
from typing import Optional

class CosyVoiceStreamer:
    """CosyVoice专业流式播放器"""
    
    def __init__(self, 
                 server_url: str = "http://localhost:9234",
                 sample_rate: int = 22050,
                 buffer_size: int = 1024,
                 max_buffer_size: int = 10):
        """
        初始化流式播放器
        
        Args:
            server_url: CosyVoice服务器地址
            sample_rate: 音频采样率
            buffer_size: 缓冲区大小
            max_buffer_size: 最大缓冲队列长度
        """
        self.server_url = server_url
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.max_buffer_size = max_buffer_size
        
        # 音频缓冲队列
        self.audio_queue = queue.Queue(maxsize=max_buffer_size)
        self.playing = False
        self.stop_event = threading.Event()
        
        # 播放统计
        self.total_chunks = 0
        self.played_chunks = 0
        
    def _audio_callback(self, outdata, frames, time, status):
        """音频播放回调函数"""
        if status:
            print(f"音频播放状态: {status}")
        
        try:
            # 从队列获取音频数据
            audio_chunk = self.audio_queue.get_nowait()
            
            # 确保数据长度匹配
            if len(audio_chunk) >= frames:
                outdata[:] = audio_chunk[:frames].reshape(-1, 1)
                # 如果有剩余数据，放回队列
                if len(audio_chunk) > frames:
                    remaining = audio_chunk[frames:]
                    self.audio_queue.put(remaining)
            else:
                # 数据不足，填充零
                outdata[:len(audio_chunk)] = audio_chunk.reshape(-1, 1)
                outdata[len(audio_chunk):] = 0
                
            self.played_chunks += 1
            
        except queue.Empty:
            # 队列为空，输出静音
            outdata.fill(0)
    
    def _fetch_audio_stream(self, text: str, spk_id: str = "1"):
        """获取音频流数据"""
        url = f"{self.server_url}/inference_sft"
        data = {
            "tts_text": text,
            "spk_id": spk_id,
            "stream": True,
            "format": "stream",
            "enable_smart_split": True,
            "max_text_length": 50
        }
        
        try:
            print(f"开始请求TTS服务: {text[:30]}...")
            response = requests.post(url, data=data, stream=True, timeout=120)
            response.raise_for_status()
            
            chunk_count = 0
            accumulated_data = b""
            
            for chunk in response.iter_content(chunk_size=self.buffer_size):
                if chunk and not self.stop_event.is_set():
                    accumulated_data += chunk
                    
                    # 当累积足够数据时处理
                    while len(accumulated_data) >= self.buffer_size * 2:  # 确保有足够数据
                        # 提取一个缓冲区大小的数据
                        chunk_data = accumulated_data[:self.buffer_size * 2]
                        accumulated_data = accumulated_data[self.buffer_size * 2:]
                        
                        # 转换为音频数组
                        audio_data = np.frombuffer(chunk_data, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        if len(audio_data) > 0:
                            # 添加到播放队列
                            try:
                                self.audio_queue.put(audio_data, timeout=1.0)
                                chunk_count += 1
                                self.total_chunks += 1
                                
                                if chunk_count % 10 == 0:
                                    print(f"已接收 {chunk_count} 个音频块...")
                                    
                            except queue.Full:
                                print("音频缓冲队列已满，跳过数据块")
                                
            # 处理剩余数据
            if accumulated_data and not self.stop_event.is_set():
                audio_data = np.frombuffer(accumulated_data, dtype=np.int16).astype(np.float32) / 32768.0
                if len(audio_data) > 0:
                    try:
                        self.audio_queue.put(audio_data, timeout=1.0)
                        self.total_chunks += 1
                    except queue.Full:
                        pass
                        
            print(f"音频流接收完成，共 {chunk_count} 个块")
            
        except requests.exceptions.RequestException as e:
            print(f"网络请求错误: {e}")
        except Exception as e:
            print(f"音频流处理错误: {e}")
    
    def play_stream(self, text: str, spk_id: str = "1", volume: float = 1.0):
        """
        播放流式TTS音频
        
        Args:
            text: 要合成的文本
            spk_id: 说话人ID
            volume: 音量 (0.0-1.0)
        """
        if self.playing:
            print("已有音频在播放中，请先停止")
            return
            
        self.playing = True
        self.stop_event.clear()
        self.total_chunks = 0
        self.played_chunks = 0
        
        print(f"开始流式播放: {text}")
        print(f"说话人: {spk_id}, 音量: {volume}")
        
        try:
            # 启动音频获取线程
            fetch_thread = threading.Thread(
                target=self._fetch_audio_stream, 
                args=(text, spk_id)
            )
            fetch_thread.daemon = True
            fetch_thread.start()
            
            # 等待一些数据到达
            print("等待音频数据...")
            while self.audio_queue.empty() and fetch_thread.is_alive():
                time.sleep(0.1)
            
            if not self.audio_queue.empty():
                print("开始播放音频...")
                
                # 开始音频播放
                with sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype=np.float32,
                    callback=self._audio_callback,
                    blocksize=self.buffer_size
                ):
                    # 等待播放完成
                    while (fetch_thread.is_alive() or not self.audio_queue.empty()) and not self.stop_event.is_set():
                        time.sleep(0.1)
                        
                        # 显示播放进度
                        if self.total_chunks > 0:
                            progress = (self.played_chunks / self.total_chunks) * 100
                            print(f"\r播放进度: {progress:.1f}% ({self.played_chunks}/{self.total_chunks})", end="")
                
                print(f"\n播放完成！")
            else:
                print("未接收到音频数据")
                
        except Exception as e:
            print(f"播放错误: {e}")
        finally:
            self.playing = False
            self.stop_event.set()
    
    def stop(self):
        """停止播放"""
        if self.playing:
            print("停止播放...")
            self.stop_event.set()
            self.playing = False
    
    def play_wav_simple(self, text: str, spk_id: str = "1"):
        """
        简单WAV播放模式（非流式，但更稳定）
        
        Args:
            text: 要合成的文本
            spk_id: 说话人ID
        """
        url = f"{self.server_url}/inference_sft"
        data = {
            "tts_text": text,
            "spk_id": spk_id,
            "stream": False,
            "format": "wav"
        }
        
        try:
            print(f"请求TTS合成: {text}")
            response = requests.post(url, data=data, timeout=60)
            response.raise_for_status()
            
            # 解析WAV数据
            wav_data = io.BytesIO(response.content)
            with wave.open(wav_data, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                sample_rate = wav_file.getframerate()
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            
            print(f"播放音频 (长度: {len(audio_data)/sample_rate:.2f}秒)")
            
            # 播放音频
            sd.play(audio_data, sample_rate)
            sd.wait()  # 等待播放完成
            
            print("播放完成！")
            
        except requests.exceptions.RequestException as e:
            print(f"网络请求错误: {e}")
        except Exception as e:
            print(f"播放错误: {e}")


def main():
    """主函数 - 交互式TTS播放器"""
    print("=" * 60)
    print("🎵 CosyVoice 专业流式音频播放器")
    print("=" * 60)
    
    # 创建播放器实例
    player = CosyVoiceStreamer()
    
    # 检查sounddevice是否正常工作
    try:
        print("检查音频设备...")
        devices = sd.query_devices()
        print(f"找到 {len(devices)} 个音频设备")
        print(f"默认输出设备: {sd.query_devices(kind='output')['name']}")
    except Exception as e:
        print(f"音频设备检查失败: {e}")
        return
    
    print("\n使用说明:")
    print("1. 输入文本进行流式播放")
    print("2. 输入 'simple:文本' 进行简单WAV播放")
    print("3. 输入 'quit' 退出")
    print("4. 播放过程中按 Ctrl+C 可以中断")
    
    try:
        while True:
            print("\n" + "-" * 40)
            text = input("请输入要合成的文本: ").strip()
            
            if not text:
                continue
                
            if text.lower() == 'quit':
                break
                
            if text.startswith('simple:'):
                # 简单WAV播放模式
                actual_text = text[7:].strip()
                if actual_text:
                    player.play_wav_simple(actual_text)
            else:
                # 流式播放模式
                try:
                    player.play_stream(text)
                except KeyboardInterrupt:
                    print("\n用户中断播放")
                    player.stop()
                    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        player.stop()
        print("感谢使用！")


if __name__ == "__main__":
    # 检查依赖
    try:
        import sounddevice as sd
        import requests
        import numpy as np
        import wave
    except ImportError as e:
        print(f"缺少依赖库: {e}")
        print("请安装: pip install sounddevice requests numpy")
        exit(1)
    
    main()
