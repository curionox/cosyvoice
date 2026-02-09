import os
import sys
import torch
import torchaudio
import librosa

lower_sr = 16000
high_sr = 22050

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    """裁剪、归一化、补零后返回音频 Tensor"""
    max_val = 0.8
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    zeros = torch.zeros(1, int(high_sr * 0.2))
    speech = torch.concat([speech, zeros], dim=1)
    return speech

def load_spk_from_wav(wav_file, cosyvoice):
    """从 wav 文件提取说话人特征，返回 dict"""
    target_wav, sample_rate = torchaudio.load(wav_file)
    if target_wav.shape[0] == 2:
        target_wav = target_wav.mean(dim=0, keepdim=True)  # 双声道转单声道

    target_wav_high = torchaudio.transforms.Resample(sample_rate, high_sr)(target_wav)
    target_wav_high = postprocess(target_wav_high)
    target_wav_lower = torchaudio.transforms.Resample(high_sr, lower_sr)(target_wav_high)

    speech_feat, speech_feat_len = cosyvoice.frontend._extract_speech_feat(target_wav_high)
    speech_token, speech_token_len = cosyvoice.frontend._extract_speech_token(target_wav_lower)
    embedding = cosyvoice.frontend._extract_spk_embedding(target_wav_lower)

    return {
        "speech_feat": speech_feat.cpu(),
        "speech_feat_len": speech_feat_len,
        "speech_token": speech_token.cpu(),
        "speech_token_len": speech_token_len,
        "embedding": embedding.cpu()
    }

def save_spk_to_pt(spk_id, spk_data, spk_dir="./speakers"):
    """保存说话人特征为 .pt 文件"""
    os.makedirs(spk_dir, exist_ok=True)
    pt_path = os.path.join(spk_dir, f"{spk_id}.pt")
    torch.save(spk_data, pt_path)
    print(f"保存特征: {pt_path}")

def load_spk_from_pt(spk_id, spk_dir="./speakers"):
    """加载已保存的 .pt 说话人特征"""
    spk_pt = os.path.join(spk_dir, f"{spk_id}.pt")
    if os.path.exists(spk_pt) and os.path.isfile(spk_pt):
        return torch.load(spk_pt)
    return None

def scan_spks_from_file(spk_dir="./speakers"):
    """扫描所有 .pt 说话人文件，返回 spk_id 列表"""
    spks = []
    for spk_pt in os.listdir(spk_dir):
        if spk_pt.endswith('.pt'):
            spks.append(spk_pt.replace(".pt", ""))
    return spks

def batch_extract_from_wavs(wav_root, cosyvoice, spk_dir="./speakers"):
    """批量遍历 wav_root 下所有 wav，提取并保存 .pt"""
    for root, dirs, files in os.walk(wav_root):
        for fname in files:
            if fname.endswith(".wav"):
                wav_path = os.path.join(root, fname)
                spk_id = os.path.splitext(fname)[0]  # 可自定义，如用父目录名
                print(f"处理: {spk_id}, 文件: {wav_path}")
                spk_data = load_spk_from_wav(wav_path, cosyvoice)
                save_spk_to_pt(spk_id, spk_data, spk_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", type=str, default="./wavs", help="音频输入目录")
    parser.add_argument("--spk_dir", type=str, default="./speakers", help=".pt输出目录")
    parser.add_argument("--model_dir", type=str, default="pretrained_models/CosyVoice2-0.5B", help="CosyVoice2模型目录")
    parser.add_argument("--mode", type=str, choices=["extract", "scan"], default="extract", help="extract:批量提取; scan:列出所有说话人")
    args = parser.parse_args()

    # 初始化 CosyVoice2 实例
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from cosyvoice.cli.cosyvoice import CosyVoice2
    cosyvoice = CosyVoice2(args.model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False)

    if args.mode == "extract":
        batch_extract_from_wavs(args.wav_dir, cosyvoice, args.spk_dir)
    elif args.mode == "scan":
        spks = scan_spks_from_file(args.spk_dir)
        print("检测到的说话人：", spks)