import os
import sys
import fnmatch
import argparse
import torch
import torchaudio
import librosa

# Add necessary paths (based on official setup)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
sys.path.append(f'{ROOT_DIR}/third_party/AcademiCodec')
sys.path.append(f'{ROOT_DIR}/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

max_val = 0.8
prompt_sr, target_sr = 16000, 22050  # Adjust if needed for CosyVoice2-0.5B

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech

class Trainer:
    def __init__(self, model_dir):
        self.cosyvoice = CosyVoice2(model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
        self.model_dir = model_dir

    def ensure_embedding(self, spk_name, prompt_speech_16k):
        """确保 spk2info 有 embedding 字段，否则补全"""
        spkinfo = self.cosyvoice.frontend.spk2info.get(spk_name, {})
        need_save = False

        # 检查 llm_embedding, flow_embedding 字段
        if 'llm_embedding' not in spkinfo or spkinfo['llm_embedding'] is None:
            print(f"[补全] {spk_name} 缺少 'llm_embedding', 自动生成...")
            # CosyVoice2 可能有类似接口，实际请查源码
            if hasattr(self.cosyvoice, "make_llm_embedding"):
                spkinfo['llm_embedding'] = self.cosyvoice.make_llm_embedding(prompt_speech_16k)
                need_save = True
            else:
                print("警告: 无法自动生成 llm_embedding, 请查阅 CosyVoice2 文档或源码！")

        if 'flow_embedding' not in spkinfo or spkinfo['flow_embedding'] is None:
            print(f"[补全] {spk_name} 缺少 'flow_embedding', 自动生成...")
            if hasattr(self.cosyvoice, "make_flow_embedding"):
                spkinfo['flow_embedding'] = self.cosyvoice.make_flow_embedding(prompt_speech_16k)
                need_save = True
            else:
                print("警告: 无法自动生成 flow_embedding, 请查阅 CosyVoice2 文档或源码！")

        if need_save:
            self.cosyvoice.frontend.spk2info[spk_name] = spkinfo

    def start(self, dest):
        pattern = "*.wav"
        for root, dirs, files in os.walk(dest):
            for filename in fnmatch.filter(files, pattern):
                spk_name = os.path.basename(root)
                if spk_name in self.cosyvoice.frontend.spk2info:
                    print(f"音色已存在，跳过: {spk_name}")
                    continue

                print(f"开始添加音色: {spk_name}")
                a = filename.replace(".wav", "").split("#")
                prompt_text = a[-1]
                audio_path = os.path.join(root, filename)
                prompt_speech_16k = postprocess(load_wav(audio_path, prompt_sr))

                # 添加 zero-shot 说话人
                self.cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech_16k, spk_name)
                print(f"添加完成: {spk_name}")

                # 补全 embedding 字段，避免推理报错
                self.ensure_embedding(spk_name, prompt_speech_16k)

                print("当前可用说话人：", self.cosyvoice.list_available_spks())

    def save(self):
        self.cosyvoice.save_spkinfo()
        spk2info_path = os.path.join(self.model_dir, 'spk2info.pt')
        spk2info = torch.load(spk2info_path)
        print("所有可用spk_id：", list(spk2info.keys()))
        for spk_id in spk2info.keys():
            print(spk_id)
            print(list(spk2info[spk_id].keys()))  # Print keys to confirm format

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--audio_dir',
                        type=str,
                        default='',
                        help='输入一个目录,目录名对应模型名。')
    args = parser.parse_args()
    
    trainer = Trainer(args.model_dir)
    trainer.start(args.audio_dir)
    trainer.save()