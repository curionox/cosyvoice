import torch
from pathlib import Path

# 检查spk2info.pt文件
spk2info_path = Path("pretrained_models/CosyVoice-300M-SFT/spk2info.pt")

if spk2info_path.exists():
    print("Loading spk2info.pt...")
    spk2info = torch.load(spk2info_path, map_location='cpu')
    
    if isinstance(spk2info, dict):
        voices = list(spk2info.keys())
        print(f"Found {len(voices)} voices:")
        for i, voice in enumerate(voices):
            print(f"{i+1}. {voice}")
    else:
        print(f"spk2info is not a dict, it's: {type(spk2info)}")
else:
    print(f"File not found: {spk2info_path}")
