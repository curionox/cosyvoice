#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将spk2info.pt文件中的音色名称转换为数字索引
"""

import torch
import json
import shutil
from pathlib import Path

def convert_spk_to_numbers():
    """将音色名称转换为数字索引"""
    
    # 文件路径
    original_file = Path("pretrained_models/CosyVoice-300M-SFT/spk2info.pt")
    backup_file = Path("pretrained_models/CosyVoice-300M-SFT/spk2info_backup.pt")
    mapping_file = Path("voice_index_mapping.json")
    
    if not original_file.exists():
        print(f"错误：找不到文件 {original_file}")
        return
    
    try:
        # 备份原文件
        print(f"备份原文件到 {backup_file}")
        shutil.copy2(original_file, backup_file)
        
        # 加载原始数据
        print(f"加载原始数据...")
        spk2info = torch.load(original_file, map_location='cpu')
        
        if not isinstance(spk2info, dict):
            print(f"错误：spk2info不是字典类型，而是: {type(spk2info)}")
            return
        
        print(f"原始音色数量: {len(spk2info)}")
        
        # 创建新的数字索引字典和映射表
        new_spk2info = {}
        index_mapping = {}
        
        # 按原始顺序转换
        for index, (original_name, voice_data) in enumerate(spk2info.items()):
            str_index = str(index)
            new_spk2info[str_index] = voice_data
            index_mapping[str_index] = original_name
            print(f"索引 {index:3d} -> {original_name}")
        
        # 保存新的pt文件
        print(f"\n保存新的pt文件...")
        torch.save(new_spk2info, original_file)
        
        # 保存映射表
        print(f"保存索引映射表到 {mapping_file}")
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(index_mapping, f, ensure_ascii=False, indent=2)
        
        print(f"\n转换完成！")
        print(f"- 原文件已备份到: {backup_file}")
        print(f"- 新的pt文件: {original_file}")
        print(f"- 索引映射表: {mapping_file}")
        print(f"- 转换了 {len(new_spk2info)} 个音色")
        
        # 显示前10个映射
        print(f"\n前10个音色的索引映射:")
        for i in range(min(10, len(index_mapping))):
            str_i = str(i)
            if str_i in index_mapping:
                print(f"  {str_i} -> {index_mapping[str_i]}")
        
        return True
        
    except Exception as e:
        print(f"转换过程中出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 如果出错，尝试恢复备份
        if backup_file.exists():
            print(f"尝试恢复备份文件...")
            shutil.copy2(backup_file, original_file)
            print(f"已恢复原文件")
        
        return False

def verify_conversion():
    """验证转换结果"""
    try:
        original_file = Path("pretrained_models/CosyVoice-300M-SFT/spk2info.pt")
        mapping_file = Path("voice_index_mapping.json")
        
        if not original_file.exists() or not mapping_file.exists():
            print("验证失败：文件不存在")
            return
        
        # 加载转换后的数据
        spk2info = torch.load(original_file, map_location='cpu')
        
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        print(f"\n验证结果:")
        print(f"- pt文件中的音色数量: {len(spk2info)}")
        print(f"- 映射表中的条目数量: {len(mapping)}")
        
        # 检查键是否都是数字
        all_numeric = all(key.isdigit() for key in spk2info.keys())
        print(f"- 所有键都是数字: {all_numeric}")
        
        # 显示一些示例
        print(f"\n示例验证:")
        for i in range(min(5, len(spk2info))):
            str_i = str(i)
            if str_i in spk2info and str_i in mapping:
                print(f"  索引 {str_i}: {mapping[str_i]} - 数据存在: {spk2info[str_i] is not None}")
        
    except Exception as e:
        print(f"验证过程中出错: {e}")

if __name__ == "__main__":
    print("=== 开始转换音色索引 ===")
    success = convert_spk_to_numbers()
    
    if success:
        print("\n=== 验证转换结果 ===")
        verify_conversion()
    else:
        print("\n转换失败！")
