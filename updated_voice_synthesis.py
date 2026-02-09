import torch
from cosyvoice.cli.cosyvoice import CosyVoice2
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'third_party', 'Matcha-TTS'))
import soundfile as sf
import numpy as np

# 模型目录
model_dir = r"d:/kt/aivioce/CosyVoice/pretrained_models/CosyVoice2-0.5B"

# 现在可以直接使用数字作为音色ID
spk_id = "2"  # 对应"中文女"音色，您可以改成任何0-183的数字

# 初始化模型
cosyvoice = CosyVoice2(model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False)

# 加载音色信息（现在使用数字索引的pt文件）
spk_info_path = "pretrained_models/CosyVoice-300M-SFT/spk2info.pt"
spk2info = torch.load(spk_info_path)

# 注册说话人（现在直接使用数字ID）
if spk_id in spk2info:
    cosyvoice.frontend.spk2info[spk_id] = spk2info[spk_id]
    print(f"已加载音色 {spk_id}")
else:
    print(f"错误：找不到音色ID {spk_id}")
    print(f"可用的音色ID范围：0-{len(spk2info)-1}")
    exit(1)

# 要合成的文本
tts_text = "我一直在思考如何提高我的数学技能，想了解一些辅导方案。我需要一个循序渐进的方案来解决那些棘手的代数问题。我听说有些在线平台提供个性化课程，这真的能帮助我保持学习进度。也许我还可以找一位当地的辅导老师，可以在晚上下班后和我见面。问题在于找到最适合我的日程安排和学习风格的辅导方式。"

# 进行语音合成
result = list(cosyvoice.inference_sft(tts_text, spk_id))

# 保存为 wav 文件
output_path = f"output_voice_{spk_id}.wav"
print("合成结果:", result)
print("音频数据:", result[0])

# 保存音频
audio_data = result[0]['tts_speech'].cpu().numpy()
if audio_data.ndim > 1:
    audio_data = audio_data.squeeze()  # 移除多余的维度

sf.write(output_path, audio_data, 22050)
print(f"已保存合成语音到 {output_path}")
print(f"使用的音色ID: {spk_id}")

# 显示音色对应关系（可选）
import json
try:
    with open('voice_index_mapping.json', 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    if spk_id in mapping:
        print(f"音色名称: {mapping[spk_id]}")
except:
    print("无法读取音色映射文件")
