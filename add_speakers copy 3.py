#读语音
import torch
from cosyvoice.cli.cosyvoice import CosyVoice2
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'third_party', 'Matcha-TTS'))
import soundfile as sf
import numpy as np
model_dir = r"d:/kt/aivioce/CosyVoice/pretrained_models/CosyVoice2-0.5B"

spk_pt_path = "speakers/audio.pt"
spk_id = "4"  # 可以自定义，只要唯一

cosyvoice = CosyVoice2(model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False)

spk_info = torch.load(spk_pt_path)
cosyvoice.frontend.spk2info[spk_id] = spk_info  # 注册说话人

tts_text = "我一直在思考如何提高我的数学技能，想了解一些辅导方案。我需要一个循序渐进的方案来解决那些棘手的代数问题。我听说有些在线平台提供个性化课程，这真的能帮助我保持学习进度。也许我还可以找一位当地的辅导老师，可以在晚上下班后和我见面。问题在于找到最适合我的日程安排和学习风格的辅导方式。"
result = list(cosyvoice.inference_sft(tts_text, spk_id))  # 结果通常为音频流或 tensor

# 保存为 wav 文件
import torchaudio
output_path = "output.wav"
print(result)
print(result[0])
#torchaudio.save(output_path, result[0]['tts_speech'].cpu(), 22050)
audio_data = result[0]['tts_speech'].cpu().numpy()
if audio_data.ndim > 1:
    audio_data = audio_data.squeeze()  # 移除多余的维度
sf.write(output_path, audio_data, 22050)
print(f"已保存合成语音到 {output_path}")