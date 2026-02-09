#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音色查询助手 - 帮助查看数字索引对应的音色名称
"""

import json
import sys

def load_voice_mapping():
    """加载音色映射表"""
    try:
        with open('voice_index_mapping.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("错误：找不到 voice_index_mapping.json 文件")
        return None
    except Exception as e:
        print(f"加载映射文件时出错: {e}")
        return None

def show_all_voices():
    """显示所有音色"""
    mapping = load_voice_mapping()
    if not mapping:
        return
    
    print("所有可用音色:")
    print("=" * 60)
    
    # 按类别分组显示
    categories = {
        "基础音色": [],
        "云系列": [],
        "晓系列": [],
        "崩坏星穹铁道": [],
        "原神角色": [],
        "其他": []
    }
    
    for idx, name in mapping.items():
        if name in ["中文女", "中文男", "日语男", "粤语女", "英文女", "英文男", "韩语女"]:
            categories["基础音色"].append((idx, name))
        elif name.startswith("云"):
            categories["云系列"].append((idx, name))
        elif name.startswith("晓"):
            categories["晓系列"].append((idx, name))
        elif name in ["三月七", "丹恒", "佩拉", "停云", "克拉拉", "刃", "卡芙卡", "卢卡", "可可利亚", "史瓦罗", "姬子", "娜塔莎", "寒鸦", "尾巴", "布洛妮娅", "希儿", "希露瓦", "帕姆", "开拓者(女)", "开拓者(男)", "彦卿", "托帕&账账", "景元", "杰帕德", "桂乃芬", "桑博", "流萤", "玲可", "瓦尔特", "白露", "真理医生", "砂金", "符玄", "米沙", "素裳", "罗刹", "艾丝妲", "花火", "藿藿", "虎克", "螺丝咕姆", "银枝", "银狼", "镜流", "阮•梅", "阿兰", "雪衣", "青雀", "驭空", "中立", "开心", "难过", "黄泉", "黑塔", "黑天鹅"]:
            categories["崩坏星穹铁道"].append((idx, name))
        elif name in ["七七", "丽莎", "久岐忍", "九条裟罗", "云堇", "五郎", "优菈", "八重神子", "凝光", "凯亚", "凯瑟琳", "刻晴", "北斗", "卡维", "可莉", "嘉明", "坎蒂丝", "夏沃蕾", "夏洛蒂", "多莉", "夜兰", "奥兹", "妮露", "娜维娅", "安柏", "宵宫", "戴因斯雷布", "托马", "提纳里", "早柚", "林尼", "枫原万叶", "柯莱", "派蒙-兴奋说话", "派蒙-吞吞吐吐", "派蒙-平静", "派蒙-很激动", "派蒙-疑惑", "流浪者", "温迪", "烟绯", "珊瑚宫心海", "珐露珊", "班尼特", "琳妮特", "琴", "瑶瑶", "甘雨", "申鹤", "白术", "砂糖", "神里绫人", "神里绫华", "空", "米卡", "纳西妲", "绮良良", "罗莎莉亚", "胡桃", "艾尔海森", "芭芭拉", "荒泷一斗", "荧", "莫娜", "莱依拉", "莱欧斯利", "菲米尼", "菲谢尔", "萍姥姥", "行秋", "诺艾尔", "赛诺", "辛焱", "达达利亚", "迪卢克", "迪奥娜", "迪娜泽黛", "迪希雅", "那维莱特", "重云", "钟离", "闲云", "阿贝多", "雷泽", "雷电将军", "香菱", "魈", "鹿野院平藏"]:
            categories["原神角色"].append((idx, name))
        else:
            categories["其他"].append((idx, name))
    
    for category, voices in categories.items():
        if voices:
            print(f"\n{category}:")
            print("-" * 30)
            for idx, name in sorted(voices, key=lambda x: int(x[0])):
                print(f"  {idx:3s} -> {name}")

def search_voice(query):
    """搜索音色"""
    mapping = load_voice_mapping()
    if not mapping:
        return
    
    query = query.lower()
    results = []
    
    for idx, name in mapping.items():
        if query in name.lower() or query in idx:
            results.append((idx, name))
    
    if results:
        print(f"搜索 '{query}' 的结果:")
        print("-" * 30)
        for idx, name in sorted(results, key=lambda x: int(x[0])):
            print(f"  {idx:3s} -> {name}")
    else:
        print(f"没有找到包含 '{query}' 的音色")

def get_voice_name(index):
    """根据索引获取音色名称"""
    mapping = load_voice_mapping()
    if not mapping:
        return
    
    if str(index) in mapping:
        print(f"音色 {index}: {mapping[str(index)]}")
    else:
        print(f"错误：找不到索引 {index}")
        print(f"可用索引范围: 0-{len(mapping)-1}")

def main():
    if len(sys.argv) < 2:
        print("音色查询助手")
        print("用法:")
        print("  python voice_helper.py all          # 显示所有音色")
        print("  python voice_helper.py search 中文   # 搜索包含'中文'的音色")
        print("  python voice_helper.py get 2        # 获取索引2对应的音色名称")
        return
    
    command = sys.argv[1].lower()
    
    if command == "all":
        show_all_voices()
    elif command == "search" and len(sys.argv) > 2:
        search_voice(sys.argv[2])
    elif command == "get" and len(sys.argv) > 2:
        try:
            index = int(sys.argv[2])
            get_voice_name(index)
        except ValueError:
            print("错误：索引必须是数字")
    else:
        print("无效的命令或参数")

if __name__ == "__main__":
    main()
