#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取spk2info.pt文件中的音色信息
"""

import torch
import sys
from pathlib import Path

def read_spk2info():
    """读取并显示spk2info.pt文件中的音色信息"""
    spk2info_path = Path("pretrained_models/CosyVoice-300M-SFT/spk2info.pt")
    
    if not spk2info_path.exists():
        print(f"错误：找不到文件 {spk2info_path}")
        return
    
    try:
        print(f"正在加载 {spk2info_path}...")
        spk2info = torch.load(spk2info_path, map_location='cpu')
        
        print(f"音色信息类型: {type(spk2info)}")
        
        if isinstance(spk2info, dict):
            print(f"可用音色数量: {len(spk2info)}")
            print("\n所有可用音色列表:")
            print("=" * 50)
            for i, spk_id in enumerate(spk2info.keys(), 1):
                print(f"{i:3d}. {spk_id}")
                
            print("\n" + "=" * 50)
            print("前3个音色的详细信息:")
            for i, (spk_id, info) in enumerate(list(spk2info.items())[:3]):
                print(f"\n音色 {i+1}: {spk_id}")
                if isinstance(info, dict):
                    for key, value in info.items():
                        if hasattr(value, 'shape'):
                            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                        else:
                            print(f"  {key}: {type(value)}")
                else:
                    print(f"  类型: {type(info)}")
        else:
            print(f"spk2info不是字典类型，而是: {type(spk2info)}")
            
    except Exception as e:
        print(f"加载文件时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    read_spk2info()
