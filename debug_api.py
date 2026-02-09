#!/usr/bin/env python3
import requests
import json

def test_api_voices():
    base_url = "http://127.0.0.1:9233"
    
    # 先获取可用音色
    try:
        response = requests.get(f"{base_url}/voices")
        voices_data = response.json()
        print("可用音色:")
        voices = voices_data['data']['voices'][:3]  # 取前3个
        for i, voice in enumerate(voices):
            print(f"{i+1}. {voice}")
        print()
    except Exception as e:
        print(f"获取音色列表失败: {e}")
        return
    
    # 测试不同音色
    text = "这是一个测试语音合成的句子"
    
    for i, voice in enumerate(voices):
        print(f"测试音色 {i+1}: {voice}")
        try:
            # 使用不同的seed确保随机性
            params = {
                'text': text,
                'role': voice,
                'seed': 1000 + i  # 每个音色使用不同seed
            }
            
            response = requests.post(f"{base_url}/tts", data=params)
            
            if response.status_code == 200:
                # 保存音频文件用于对比
                filename = f"test_voice_{i+1}_{voice.replace(',', '_').replace(' ', '_')}.wav"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"  [OK] 生成成功: {filename} (大小: {len(response.content)} bytes)")
                
                # 检查响应头
                if 'X-Voice-Warning' in response.headers:
                    print(f"  [WARN] 警告: {response.headers['X-Voice-Warning']}")
                    print(f"  原始音色: {response.headers.get('X-Original-Voice')}")
                    print(f"  使用音色: {response.headers.get('X-Used-Voice')}")
            else:
                print(f"  [ERROR] 请求失败: {response.status_code}")
                print(f"  错误: {response.text}")
                
        except Exception as e:
            print(f"  [ERROR] 异常: {e}")
        print()

if __name__ == "__main__":
    test_api_voices()