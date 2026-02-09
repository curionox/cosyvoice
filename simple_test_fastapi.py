#!/usr/bin/env python3
import requests

def simple_test():
    base_url = "http://localhost:8235"
    
    # 测试林尼音色
    print("测试林尼音色...")
    data = {
        "tts_text": "你好啊亲爱的朋友们",
        "spk_id": "林尼",
        "format": "wav"
    }
    
    try:
        response = requests.post(f"{base_url}/inference_sft", data=data)
        
        if response.status_code == 200:
            filename = "fastapi_linni.wav"
            with open(filename, "wb") as f:
                f.write(response.content)
            
            print(f"成功生成: {filename}")
            print(f"文件大小: {len(response.content)} bytes")
        else:
            print(f"请求失败: {response.status_code}")
            print(f"错误: {response.text}")
            
    except Exception as e:
        print(f"异常: {e}")

    # 测试真理医生音色
    print("\n测试真理医生音色...")
    data = {
        "tts_text": "你好啊亲爱的朋友们", 
        "spk_id": "真理医生",
        "format": "wav"
    }
    
    try:
        response = requests.post(f"{base_url}/inference_sft", data=data)
        
        if response.status_code == 200:
            filename = "fastapi_doctor.wav"
            with open(filename, "wb") as f:
                f.write(response.content)
            
            print(f"成功生成: {filename}")
            print(f"文件大小: {len(response.content)} bytes")
        else:
            print(f"请求失败: {response.status_code}")
            print(f"错误: {response.text}")
            
    except Exception as e:
        print(f"异常: {e}")

if __name__ == "__main__":
    simple_test()