#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_cosyvoice_spk.py
改进版 c
用法示例:
  python train_cosyvoice_spk.py --model_dir D:\kt\aivioce\CosyVoice\pretrained_models\CosyVoice2-0.5B --audio_dir D:\data\voice_samples --workers 4
"""

import os
import sys
import argparse
import fnmatch
import shutil
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Tuple

# 请确保你的环境下已把 CosyVoice 路径加入 sys.path（你之前的做法）
# sys.path.insert(0, r'D:\kt\aivioce\CosyVoice')
# sys.path.append(r'D:\kt\aivioce\CosyVoice\third_party\AcademiCodec')
# sys.path.append(r'D:\kt\aivioce\CosyVoice\third_party\Matcha-TTS')

from hyperpyyaml import load_hyperpyyaml
import torch
import torchaudio
import librosa
import numpy as np

# cosyvoice imports - 保持与你环境一致的导入路径
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.utils.file_utils import load_wav

# 可配置项
MAX_VAL = 0.8
PROMPT_SR = 16000
TARGET_SR = 22050

# Audio quality thresholds (可按需调整)
MIN_SAMPLE_SECONDS = 1.0   # 最短 1 秒
MAX_SAMPLE_SECONDS = 300.0 # 最长 5 分钟（防止上传整段音频）
MIN_RMS_DB = -50.0         # 最低 RMS (dB) 检测过度静音
SILENCE_TOP_DB = 60        # librosa.trim top_db used in postprocess

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def postprocess(speech: torch.Tensor, top_db=SILENCE_TOP_DB, hop_length=220, win_length=440) -> torch.Tensor:
    """裁剪静音、归一化峰值并补 0.2s 尾点"""
    # speech: torch.Tensor shape (1, N) or (N,) -> we ensure shape (1,N)
    if speech.ndim == 1:
        speech = speech.unsqueeze(0)
    speech_np = speech.cpu().numpy().squeeze(0)
    trimmed, _ = librosa.effects.trim(speech_np, top_db=top_db, frame_length=win_length, hop_length=hop_length)
    speech = torch.from_numpy(trimmed).unsqueeze(0)
    if speech.abs().max() > 0:
        peak = float(speech.abs().max())
        if peak > MAX_VAL:
            speech = speech / peak * MAX_VAL
    # append 0.2s zeros at TARGET_SR
    pad_len = int(TARGET_SR * 0.2)
    pad = torch.zeros((1, pad_len), dtype=speech.dtype)
    speech = torch.cat([speech, pad], dim=1)
    return speech


def _rms_db(wav: np.ndarray) -> float:
    """Return RMS in dB for numpy samples (mono)"""
    eps = 1e-9
    rms = np.sqrt(np.mean(wav ** 2) + eps)
    db = 20 * np.log10(rms + eps)
    return db


class Trainer(CosyVoiceFrontEnd):
    def __init__(self, model_dir: str, config_path: str):
        self.model_dir = str(model_dir)
        # load cosyvoice config
        with open(config_path, 'r', encoding='utf-8') as f:
            configs = load_hyperpyyaml(f)
        # init parent; keep instruct=False to match你的原逻辑
        instruct = False
        super().__init__(
            configs['get_tokenizer'],
            configs['feat_extractor'],
            f'{self.model_dir}/campplus.onnx',
            f'{self.model_dir}/speech_tokenizer_v1.onnx',
            f'{self.model_dir}/spk2info.pt',
            instruct,
            configs['allowed_special']
        )

        # ensure self.spk2info exists (parent may load or create)
        if not hasattr(self, 'spk2info') or self.spk2info is None:
            self.spk2info = {}

    def validate_and_prepare(self, file_path: str) -> Tuple[bool, str, torch.Tensor]:
        """
        Validate audio: load, check duration, RMS, convert to PROMPT_SR and return the tensor (1,N)
        返回: (ok, reason, speech_tensor_16k)
        """
        try:
            speech = load_wav(file_path, PROMPT_SR)  # 使用 cosyvoice 提供 loader，返回 torch.FloatTensor (1,N)
            if speech is None or speech.numel() == 0:
                return False, "empty_audio", None
            # ensure mono
            if speech.ndim == 2 and speech.shape[0] > 1:
                # mixdown
                speech = torch.mean(speech, dim=0, keepdim=True)
            duration = speech.shape[-1] / PROMPT_SR
            if duration < MIN_SAMPLE_SECONDS:
                return False, f"too_short:{duration:.2f}s", speech
            if duration > MAX_SAMPLE_SECONDS:
                return False, f"too_long:{duration:.2f}s", speech
            # compute RMS dB
            wav_np = speech.cpu().numpy().squeeze(0)
            rms_db = _rms_db(wav_np)
            if rms_db < MIN_RMS_DB:
                return False, f"too_quiet_rms_db:{rms_db:.1f}", speech
            return True, "ok", speech
        except Exception as e:
            return False, f"load_error:{e}", None

    def process_sample_file(self, file_path: str, spk_name: str, text_from_filename: str, force: bool = False) -> Tuple[str, bool, str]:
        """
        处理单个 sample.wav 文件并把向量写入 self.spk2info（内存）
        返回 (spk_name, success_bool, message)
        """
        try:
            if (spk_name in self.spk2info) and (not force):
                return spk_name, False, "exists"
            ok, reason, speech_16k = self.validate_and_prepare(file_path)
            if not ok:
                return spk_name, False, f"validate_fail:{reason}"
            # postprocess & resample
            prompt_speech_16k = postprocess(speech_16k, top_db=SILENCE_TOP_DB)
            # text token extraction from filename
            prompt_text_token, prompt_text_token_len = self._extract_text_token(text_from_filename)
            # resample to target sr for feature extraction
            prompt_speech_22050 = torchaudio.transforms.Resample(orig_freq=PROMPT_SR, new_freq=TARGET_SR)(prompt_speech_16k)
            # feature & token extract
            speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_22050)
            speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
            embedding = self._extract_spk_embedding(prompt_speech_16k)

            # Write into spk2info structure (keep in-memory)
            self.spk2info[spk_name] = {
                "embedding": embedding,
                "speech_feat": speech_feat,
                "speech_token": speech_token
            }
            return spk_name, True, "ok"
        except Exception as e:
            tb = traceback.format_exc()
            return spk_name, False, f"error:{e}\n{tb}"

    def save(self, backup: bool = True):
        """原子性保存 spk2info.pt，若 backup=True，先备份旧文件."""
        target_path = os.path.join(self.model_dir, "spk2info.pt")
        tmp_path = os.path.join(self.model_dir, "spk2info.pt.tmp")
        bak_path = os.path.join(self.model_dir, "spk2info.pt.bak")
        try:
            # backup if exists
            if backup and os.path.exists(target_path):
                shutil.copy2(target_path, bak_path)
            # save temp then move
            torch.save(self.spk2info, tmp_path)
            os.replace(tmp_path, target_path)
            print(f"[save] saved spk2info to {target_path}")
        except Exception as e:
            print(f"[save_error] failed to save spk2info: {e}")
            raise


def discover_samples(base_dir: str, pattern: str = "*.wav"):
    """遍历 base_dir，返回列表 (spk_name, file_path, text_token)"""
    samples = []
    base_dir = str(base_dir)
    for root, dirs, files in os.walk(base_dir):
        for filename in fnmatch.filter(files, pattern):
            rel_dir = os.path.relpath(root, base_dir)
            # use directory name as speaker name; 若相对路径包含多层，取最后一层
            spk_name = os.path.basename(root)
            # 原脚本里，你用 filename 不含后缀的最后部分作为文本 token
            token_candidate = filename.replace(".wav", "")
            # 如果你的 filename 含有 "name#text" 格式，提取最后一段作为文本
            if "#" in token_candidate:
                text_from_filename = token_candidate.split("#")[-1]
            else:
                text_from_filename = token_candidate
            samples.append((spk_name, os.path.join(root, filename), text_from_filename))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                        help='CosyVoice 模型目录，示例: D:\\kt\\aivioce\\CosyVoice\\pretrained_models\\CosyVoice2-0.5B')
    parser.add_argument('--config', type=str, default=r'D:\kt\aivioce\CosyVoice\examples\libritts\cosyvoice\conf\cosyvoice.yaml',
                        help='cosyvoice 配置文件路径（yaml）')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='输入目录，目录名对应说话人名称，内部包含 .wav 文件')
    parser.add_argument('--workers', type=int, default=1,
                        help='并发 worker 数（线程池），默认 1')
    parser.add_argument('--force', action='store_true',
                        help='覆盖已存在的说话人 embedding')
    parser.add_argument('--no-save', action='store_true',
                        help='处理完毕后不保存到 spk2info.pt（调试用途）')
    args = parser.parse_args()

    model_dir = args.model_dir
    audio_dir = args.audio_dir
    config_path = args.config
    workers = max(1, int(args.workers))
    force = args.force
    no_save = args.no_save

    # instantiate Trainer
    print(f"[init] model_dir={model_dir}, config={config_path}")
    trainer = Trainer(model_dir, config_path)

    print(f"[discover] scanning {audio_dir} ...")
    samples = discover_samples(audio_dir)
    if not samples:
        print("[discover] 未找到任何 wav 文件，检查路径或文件名格式。")
        return

    # deduplicate by speaker if multiple files exist; we will pick first valid file per speaker unless --force
    grouped = {}
    for spk_name, file_path, text_token in samples:
        grouped.setdefault(spk_name, []).append((file_path, text_token))

    # prepare tasks: for each speaker, take first file that validates
    tasks = []
    for spk_name, file_entries in grouped.items():
        # if speaker exists and not force, skip
        if spk_name in trainer.spk2info and not force:
            print(f"[skip] speaker already exists and --force not set: {spk_name}")
            continue
        # try to find a valid sample among the files for this speaker
        chosen = None
        for fp, tk in file_entries:
            ok, reason, _ = trainer.validate_and_prepare(fp)
            if ok:
                chosen = (spk_name, fp, tk)
                break
            else:
                # keep trying other files, but log reason
                print(f"[validate] {spk_name}:{fp} -> {reason}")
        if chosen:
            tasks.append(chosen)
        else:
            print(f"[warn] no valid sample found for speaker {spk_name}, skipped.")

    total = len(tasks)
    print(f"[plan] {total} speakers to process (workers={workers}, force={force})")

    errors = []
    successes = []

    if workers <= 1:
        iterable = tasks
        if tqdm:
            iterable = tqdm(tasks, desc="Processing speakers")
        for spk_name, fp, tk in iterable:
            name, ok, msg = trainer.process_sample_file(fp, spk_name, tk, force=force)
            if ok:
                successes.append(name)
                print(f"[ok] {name}")
            else:
                errors.append((spk_name, msg))
                print(f"[err] {spk_name} -> {msg}")
    else:
        # use ThreadPoolExecutor for concurrency
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(trainer.process_sample_file, fp, spk_name, tk, force): (spk_name, fp)
                       for spk_name, fp, tk in tasks}
            if tqdm:
                pbar = tqdm(total=len(futures), desc="Processing speakers")
            for fut in as_completed(futures):
                spk_name, fp = futures[fut]
                try:
                    name, ok, msg = fut.result()
                    if ok:
                        successes.append(name)
                        print(f"[ok] {name}")
                    else:
                        errors.append((spk_name, msg))
                        print(f"[err] {spk_name} -> {msg}")
                except Exception as e:
                    tb = traceback.format_exc()
                    errors.append((spk_name, f"exception:{e}\n{tb}"))
                    print(f"[exception] {spk_name} -> {e}")
                if tqdm:
                    pbar.update(1)
            if tqdm:
                pbar.close()

    print(f"[done] success_count={len(successes)}, fail_count={len(errors)}")
    if errors:
        print("[errors]")
        for spk, msg in errors[:50]:
            print(f" - {spk}: {msg}")

    if not no_save:
        try:
            trainer.save()
        except Exception as e:
            print(f"[save_error] {e}")
    else:
        print("[no-save] skipping saving spk2info.pt as requested (--no-save)")

    print("[finished]")

if __name__ == "__main__":
    main()
