#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接检查spk2info.pt文件中的音色信息
"""

import torch
import sys
from pathlib import Path

def check_spk2info():
    """检查spk2info.pt文件中的音色信息"""
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
            print("\n可用音色列表:")
            for i, spk_id in enumerate(spk2info.keys(), 1):
                print(f"{i:2d}. {spk_id}")
                
            print(f"\n前几个音色的详细信息:")
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

def check_with_cosyvoice():
    """使用CosyVoice类检查音色"""
    try:
        sys.path.append('.')
        from cosyvoice.cli.cosyvoice import CosyVoice
        
        model_dir = "pretrained_models/CosyVoice-300M-SFT"
        if not Path(model_dir).exists():
            print(f"错误：找不到模型目录 {model_dir}")
            return
            
        print(f"\n正在加载CosyVoice模型...")
        model = CosyVoice(model_dir, load_jit=False, fp16=False)
        
        print("获取可用音色列表...")
        available_spks = model.list_available_spks()
        
        print(f"通过CosyVoice获取的音色数量: {len(available_spks)}")
        print("\n可用音色列表:")
        for i, spk in enumerate(available_spks, 1):
            print(f"{i:2d}. {spk}")
            
    except Exception as e:
        print(f"使用CosyVoice检查时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== 检查spk2info.pt文件 ===")
    check_spk2info()
    
    print("\n" + "="*50)
    print("=== 使用CosyVoice类检查 ===")
    check_with_cosyvoice()
