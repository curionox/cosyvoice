#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CosyVoice1 spk2info.pt → CosyVoice3 spk2info.pt 完整转换工具。

核心原理：
  CosyVoice3 不支持 SFT 模式（LLM 不使用 speaker embedding），
  只能走 zero-shot 路径。所以 spk2info 必须包含完整的：
    prompt_text, speech_token, speech_feat, embedding

  转换步骤：
  1. 用 CosyVoice1 的 HiFi-GAN 把 v1 spk2info 中的 speech_feat(mel) 还原成 wav
  2. 用 CosyVoice3 的前端从 wav 重新提取 v3 格式的 speech_token + speech_feat + embedding
  3. 生成一个通用的 prompt_text（用 v3 的 tokenizer）
  4. 保存完整的 v3 spk2info.pt

使用：
  python convert_v1_to_v3_spk2info.py \
      --v1_model_dir pretrained_models/CosyVoice-300M \
      --v3_model_dir pretrained_models/Fun-CosyVoice3-0.5B
"""

import os
import sys
import argparse
import functools

# Force unbuffered stdout for real-time output
print = functools.partial(print, flush=True)

import torch
import torchaudio
import numpy as np
import whisper
import torchaudio.compliance.kaldi as kaldi

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'third_party', 'Matcha-TTS'))


def mel_to_wav_v1(speech_feat, hift_model):
    """用 CosyVoice1 的 HiFi-GAN 从 mel 还原音频。
    speech_feat: (1, T, 80)
    返回: (1, samples) 的 wav tensor, sample_rate=22050
    """
    with torch.no_grad():
        # HiFTGenerator.inference 接收 (batch, 80, T) 格式
        mel = speech_feat.transpose(1, 2)  # (1, 80, T)
        wav, _ = hift_model.inference(speech_feat=mel)
    return wav  # (1, samples)


def convert(v1_model_dir, v3_model_dir):
    """完整转换 v1 spk2info → v3 spk2info"""
    from hyperpyyaml import load_hyperpyyaml
    from cosyvoice.cli.cosyvoice import CosyVoice3

    # === Step 1: 加载 v1 的 HiFi-GAN 用于 mel → wav ===
    print("=" * 60)
    print("Step 1: 加载 CosyVoice1 的 HiFi-GAN")
    print("=" * 60)

    v1_yaml = os.path.join(v1_model_dir, 'cosyvoice.yaml')
    with open(v1_yaml, 'r') as f:
        v1_configs = load_hyperpyyaml(f)

    v1_hift = v1_configs['hift']
    v1_hift_state = torch.load(
        os.path.join(v1_model_dir, 'hift.pt'),
        map_location='cpu'
    )
    v1_hift.load_state_dict(v1_hift_state)
    v1_hift.eval()
    v1_sample_rate = v1_configs['sample_rate']  # 22050
    print(f"  v1 HiFi-GAN 加载完成, sample_rate={v1_sample_rate}")

    # === Step 2: 加载 v1 spk2info ===
    print("\nStep 2: 加载 v1 spk2info")
    v1_spk2info_path = os.path.join(v1_model_dir, 'spk2info.pt')
    v1_spk2info = torch.load(v1_spk2info_path, map_location='cpu', weights_only=True)
    print(f"  v1 说话人数量: {len(v1_spk2info)}")

    # === Step 3: 加载 CosyVoice3 模型 ===
    print("\nStep 3: 加载 CosyVoice3 模型")
    cosyvoice3 = CosyVoice3(v3_model_dir)
    v3_sample_rate = cosyvoice3.sample_rate  # 24000
    print(f"  v3 模型加载完成, sample_rate={v3_sample_rate}")

    # === Step 4: 逐个转换说话人 ===
    print("\n" + "=" * 60)
    print("Step 4: 逐个转换说话人")
    print("=" * 60)

    # 通用 prompt_text，包含 <|endofprompt|> 标记
    prompt_text = "You are a helpful assistant.<|endofprompt|>这是一段参考语音。"

    v3_spk2info = {}
    failed = []

    for idx, (spk_id, v1_info) in enumerate(v1_spk2info.items()):
        print(f"\n[{idx + 1}/{len(v1_spk2info)}] 转换说话人: {spk_id}")

        if 'speech_feat' not in v1_info:
            print(f"  [跳过] 缺少 speech_feat")
            failed.append((spk_id, "缺少 speech_feat"))
            continue

        try:
            # 4a. mel → wav (用 v1 HiFi-GAN)
            speech_feat = v1_info['speech_feat']  # (1, T, 80)
            wav_22k = mel_to_wav_v1(speech_feat, v1_hift)  # (1, samples) @ 22050
            duration = wav_22k.shape[1] / v1_sample_rate
            print(f"  mel→wav: {duration:.2f}s @ {v1_sample_rate}Hz")

            # 4b. 保存临时 wav 文件（各提取函数需要文件路径或 tensor）
            # _extract_speech_token 需要 16kHz
            # _extract_speech_feat 需要 24kHz
            # _extract_spk_embedding 需要 16kHz
            wav_16k = torchaudio.transforms.Resample(v1_sample_rate, 16000)(wav_22k)
            wav_24k = torchaudio.transforms.Resample(v1_sample_rate, 24000)(wav_22k)

            # 4c. 用 CosyVoice3 前端提取特征（直接使用 tensor，跳过 load_wav）
            frontend = cosyvoice3.frontend
            device = frontend.device

            # speech_token (v3 tokenizer, 6561 词表) — 需要 16kHz tensor
            assert wav_16k.shape[1] / 16000 <= 30, 'audio longer than 30s'
            feat = whisper.log_mel_spectrogram(wav_16k, n_mels=128)
            speech_token = frontend.speech_tokenizer_session.run(None, {
                frontend.speech_tokenizer_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                frontend.speech_tokenizer_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)
            })[0].flatten().tolist()
            speech_token = torch.tensor([speech_token], dtype=torch.int32).to(device)
            speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(device)
            print(f"  speech_token: shape={speech_token.shape}")

            # speech_feat (v3 mel 参数, 24kHz) — 需要 24kHz tensor
            speech_feat_v3 = frontend.feat_extractor(wav_24k).squeeze(dim=0).transpose(0, 1).to(device)
            speech_feat_v3 = speech_feat_v3.unsqueeze(dim=0)
            speech_feat_v3_len = torch.tensor([speech_feat_v3.shape[1]], dtype=torch.int32).to(device)
            print(f"  speech_feat: shape={speech_feat_v3.shape}")

            # embedding (campplus, 192d) — 需要 16kHz tensor
            feat_emb = kaldi.fbank(wav_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
            feat_emb = feat_emb - feat_emb.mean(dim=0, keepdim=True)
            embedding = frontend.campplus_session.run(None, {
                frontend.campplus_session.get_inputs()[0].name: feat_emb.unsqueeze(dim=0).cpu().numpy()
            })[0].flatten().tolist()
            embedding = torch.tensor([embedding]).to(device)
            print(f"  embedding: shape={embedding.shape}")

            # CosyVoice3: force speech_feat % speech_token = 2 (token_mel_ratio=2)
            token_len = min(int(speech_feat_v3.shape[1] / 2), speech_token.shape[1])
            speech_feat_v3 = speech_feat_v3[:, :2 * token_len]
            speech_feat_v3_len = torch.tensor([2 * token_len], dtype=torch.int32)
            speech_token = speech_token[:, :token_len]
            speech_token_len = torch.tensor([token_len], dtype=torch.int32)

            # prompt_text tokenize
            prompt_text_token, prompt_text_token_len = cosyvoice3.frontend._extract_text_token(prompt_text)

            # 4d. 构建完整的 v3 spk2info entry
            v3_spk2info[spk_id] = {
                'prompt_text': prompt_text_token.cpu(),
                'prompt_text_len': prompt_text_token_len.cpu(),
                'llm_prompt_speech_token': speech_token.cpu(),
                'llm_prompt_speech_token_len': speech_token_len.cpu(),
                'flow_prompt_speech_token': speech_token.cpu(),
                'flow_prompt_speech_token_len': speech_token_len.cpu(),
                'prompt_speech_feat': speech_feat_v3.cpu(),
                'prompt_speech_feat_len': speech_feat_v3_len.cpu(),
                'llm_embedding': embedding.cpu(),
                'flow_embedding': embedding.cpu(),
            }
            print(f"  [完成]")

        except Exception as e:
            print(f"  [失败] {e}")
            import traceback
            traceback.print_exc()
            failed.append((spk_id, str(e)))

    # === Step 5: 保存 ===
    print("\n" + "=" * 60)
    print("Step 5: 保存结果")
    print("=" * 60)

    output_path = os.path.join(v3_model_dir, 'spk2info.pt')
    torch.save(v3_spk2info, output_path)
    print(f"  已保存到: {output_path}")
    print(f"  成功: {len(v3_spk2info)}/{len(v1_spk2info)}")
    if failed:
        print(f"  失败: {len(failed)}")
        for spk_id, reason in failed:
            print(f"    {spk_id}: {reason}")

    # 验证
    print("\n验证第一个说话人的字段:")
    if v3_spk2info:
        first_key = list(v3_spk2info.keys())[0]
        for k, v in v3_spk2info[first_key].items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    print("\n" + "=" * 60)
    print("转换完成！")
    print(f"使用方式: inference_zero_shot 或 inference_instruct2")
    print(f"  cosyvoice3.inference_zero_shot(text, prompt_text, prompt_wav, zero_shot_spk_id='0')")
    print("=" * 60)


def inspect(path):
    """查看 spk2info 结构"""
    spk2info = torch.load(path, map_location='cpu', weights_only=True)
    print(f"文件: {path}")
    print(f"说话人数量: {len(spk2info)}")
    for i, (spk_id, info) in enumerate(list(spk2info.items())[:3]):
        print(f"\n说话人 [{i}]: {spk_id}")
        if isinstance(info, dict):
            for k, v in info.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"  {k}: {type(v)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CosyVoice1 → CosyVoice3 spk2info 转换")
    parser.add_argument('--v1_model_dir', type=str, default='pretrained_models/CosyVoice-300M',
                        help='CosyVoice1 模型目录（含 spk2info.pt 和 hift.pt）')
    parser.add_argument('--v3_model_dir', type=str, default='pretrained_models/Fun-CosyVoice3-0.5B',
                        help='CosyVoice3 模型目录')
    parser.add_argument('--mode', type=str, choices=['convert', 'inspect'], default='convert')
    parser.add_argument('--inspect_path', type=str, default='', help='inspect 模式下的文件路径')
    args = parser.parse_args()

    if args.mode == 'inspect':
        path = args.inspect_path or os.path.join(args.v1_model_dir, 'spk2info.pt')
        inspect(path)
    else:
        convert(args.v1_model_dir, args.v3_model_dir)
