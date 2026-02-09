import torch
 
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
 
model_dir = r'pretrained_models/CosyVoice2-0.5B'
cosyvoice = CosyVoice2(model_dir)
spk_id = "xijun"
audio_path = r"d:\kt\aivioce\DataVoice\1\audio.wav"
sample_text = "我我是通义实验室语音团队全新推出的生成式语音大模型。"
prompt_speech_16k = load_wav(audio_path, 16000)
cosyvoice.add_zero_shot_spk(sample_text, prompt_speech_16k, spk_id)
print("注册成功，当前可用说话人：", cosyvoice.list_available_spks())
cosyvoice.save_spkinfo()
spk2info = torch.load(model_dir + '/spk2info.pt')
print("所有可用spk_id：", list(spk2info.keys()))
for spk_id in spk2info.keys():
    print(spk_id)
    print(spk2info[spk_id].keys())
    
    
    