import requests
import pygame
import io
import numpy as np
import time

def play_tts_stream_pygame(text, spk_id="8"):
    """使用pygame的流式播放（修复版）"""
    url = "http://localhost:8234/inference_sft"
    data = {
        "tts_text": text,
        "spk_id": spk_id,
        "stream": True,
        "format": "stream"
    }
    
    try:
        print(f"开始pygame流式播放: {text}")
        response = requests.post(url, data=data, stream=True, timeout=30)
        response.raise_for_status()
        
        # 初始化pygame音频 - 修复：使用立体声模式
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
        
        chunk_count = 0
        accumulated_data = b""
        
        # 流式播放
        for chunk in response.iter_content(chunk_size=2048):
            if chunk:
                accumulated_data += chunk
                
                # 当累积足够数据时播放
                while len(accumulated_data) >= 4096:  # 确保有足够数据
                    # 提取数据块
                    chunk_data = accumulated_data[:4096]
                    accumulated_data = accumulated_data[4096:]
                    
                    # 转换为numpy数组
                    audio_data = np.frombuffer(chunk_data, dtype=np.int16)
                    
                    if len(audio_data) > 0:
                        try:
                            # 修复：转换为2维数组以支持立体声
                            # 将单声道数据复制到两个声道
                            stereo_data = np.column_stack((audio_data, audio_data))
                            
                            # 创建并播放声音
                            sound = pygame.sndarray.make_sound(stereo_data)
                            sound.play()
                            
                            # 等待播放完成
                            duration = len(audio_data) / 22050
                            time.sleep(duration * 0.8)  # 稍微重叠播放以避免间断
                            
                            chunk_count += 1
                            if chunk_count % 5 == 0:
                                print(f"已播放 {chunk_count} 个音频块...")
                                
                        except Exception as e:
                            print(f"播放音频块时出错: {e}")
                            continue
        
        # 处理剩余数据
        if accumulated_data:
            audio_data = np.frombuffer(accumulated_data, dtype=np.int16)
            if len(audio_data) > 0:
                try:
                    stereo_data = np.column_stack((audio_data, audio_data))
                    sound = pygame.sndarray.make_sound(stereo_data)
                    sound.play()
                    duration = len(audio_data) / 22050
                    time.sleep(duration)
                except Exception as e:
                    print(f"播放最后音频块时出错: {e}")
        
        print(f"pygame播放完成！共播放 {chunk_count} 个块")
        
    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {e}")
    except Exception as e:
        print(f"播放错误: {e}")
    finally:
        pygame.mixer.quit()

def play_tts_simple_pygame(text, spk_id="8"):
    """使用pygame的简单WAV播放"""
    url = "http://localhost:8234/inference_sft"
    data = {
        "tts_text": text,
        "spk_id": spk_id,
        "stream": False,
        "format": "wav"
    }
    
    try:
        print(f"开始pygame简单播放: {text}")
        response = requests.post(url, data=data, timeout=60)
        response.raise_for_status()
        
        # 初始化pygame
        pygame.mixer.init()
        
        # 直接播放WAV数据
        sound = pygame.mixer.Sound(io.BytesIO(response.content))
        sound.play()
        
        # 等待播放完成
        while pygame.mixer.get_busy():
            time.sleep(0.1)
            
        print("pygame简单播放完成！")
        
    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {e}")
    except Exception as e:
        print(f"播放错误: {e}")
    finally:
        pygame.mixer.quit()

def main():
    """主函数"""
    print("=" * 50)
    print("🎵 CosyVoice Pygame播放器（修复版）")
    print("=" * 50)
    
    print("使用说明:")
    print("1. 输入文本进行流式播放")
    print("2. 输入 'simple:文本' 进行简单WAV播放")
    print("3. 输入 'quit' 退出")
    
    try:
        while True:
            print("\n" + "-" * 30)
            text = input("请输入要合成的文本: ").strip()
            
            if not text:
                continue
                
            if text.lower() == 'quit':
                break
                
            if text.startswith('simple:'):
                # 简单WAV播放模式
                actual_text = text[7:].strip()
                if actual_text:
                    play_tts_simple_pygame(actual_text)
            else:
                # 流式播放模式
                play_tts_stream_pygame(text)
                
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        print("感谢使用！")

if __name__ == "__main__":
    main()
else:
    # 保持原有的函数调用兼容性
    def play_tts_stream(text, spk_id="8"):
        """兼容性函数"""
        return play_tts_stream_pygame(text, spk_id)
    
    # 如果直接导入使用，执行原来的示例
    if __name__ != "__main__":
        play_tts_stream("你好，这是一个测试")
