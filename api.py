import os, time, sys
from pathlib import Path
import argparse
import shutil
import logging
from logging.handlers import RotatingFileHandler
import subprocess
import datetime
import base64
import librosa
import random
from cosyvoice.utils.common import set_all_random_seed

import torch
import torchaudio
from flask import Flask, request, jsonify, send_file, make_response
#     python api.py --sft-model-path pretrained_models/CosyVoice-300M-SFT --preload-models sft
# --- Global Model Placeholders ---
#  python api.py --sft-model-path pretrained_models/CosyVoice-300M-SFT --preload-models sft
sft_model = None
tts_model = None
VOICE_LIST = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Logging Setup ---
def setup_logging(logs_dir: Path):
    log = logging.getLogger('werkzeug')
    log.handlers[:] = []
    log.setLevel(logging.WARNING)

    root_log = logging.getLogger()
    root_log.handlers = []
    root_log.setLevel(logging.WARNING)

    app.logger.setLevel(logging.WARNING)
    log_file = logs_dir / f'{datetime.datetime.now().strftime("%Y%m%d")}.log'
    file_handler = RotatingFileHandler(str(log_file), maxBytes=1024 * 1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(formatter)
    app.logger.addHandler(file_handler)

# --- Core Functions ---

def setup_environment():
    """Sets up PYTHONPATH for Matcha-TTS and validates ffmpeg availability."""
    root_dir = Path(__file__).parent
    matcha_tts_path = root_dir / 'third_party' / 'Matcha-TTS'
    if str(matcha_tts_path) not in sys.path:
        sys.path.append(str(matcha_tts_path))

    if not shutil.which("ffmpeg"):
        print("ffmpeg not found in PATH. Please ensure it is installed and accessible.")
        # Simple check for homebrew path on macOS
        if sys.platform == 'darwin' and (Path("/opt/homebrew/bin") / "ffmpeg").exists():
             os.environ["PATH"] = "/opt/homebrew/bin" + os.pathsep + os.environ["PATH"]

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg could not be found. Please install it and add it to your system's PATH.")
    print(f"ffmpeg found at: {shutil.which('ffmpeg')}")

def load_model(model_type: str, args):
    """
    Loads a specified model, downloading it if necessary and allowed.
    `model_type` can be 'sft' or 'tts'.
    """
    global sft_model, tts_model
    from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
    from modelscope import snapshot_download

    models_dir = Path(args.models_dir)

    if model_type == 'sft':
        # 优先使用自定义路径
        if hasattr(args, 'sft_model_path') and args.sft_model_path:
            local_dir = Path(args.sft_model_path)
        else:
            model_id = 'iic/CosyVoice-300M-SFT'
            local_dir = models_dir / 'CosyVoice-300M-SFT'
        if sft_model is not None: return
    elif model_type == 'tts':
        # 优先使用自定义路径
        if hasattr(args, 'tts_model_path') and args.tts_model_path:
            local_dir = Path(args.tts_model_path)
        else:
            model_id = 'iic/CosyVoice2-0.5B'
            local_dir = models_dir / 'CosyVoice2-0.5B'
        if tts_model is not None: return
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 只有在使用默认路径且模型不存在时才下载
    if not local_dir.exists() and not hasattr(args, f'{model_type}_model_path') and not args.disable_download:
        print(f"Model not found locally. Downloading {model_id} to {local_dir}...")
        snapshot_download(model_id, local_dir=str(local_dir))
    elif not local_dir.exists():
        raise FileNotFoundError(f"Model {model_type} not found at {local_dir}")

    print(f"Loading model: {model_type} from {local_dir}...")
    if model_type == 'sft':
        # 关键修复：使用与WebUI相同的加载方式，不指定额外参数
        sft_model = CosyVoice(str(local_dir))
    elif model_type == 'tts':
        # 关键修复：使用与WebUI相同的加载方式，不指定额外参数
        tts_model = CosyVoice2(str(local_dir))
    print(f"Model {model_type} loaded successfully.")

def postprocess(speech, sample_rate, top_db=60, hop_length=220, win_length=440):
    max_val = 0.8
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(sample_rate * 0.2))], dim=1)
    return speech

def base64_to_wav(encoded_str, output_path: Path):
    if not encoded_str: raise ValueError("Base64 encoded string is empty.")
    wav_bytes = base64.b64decode(encoded_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as wav_file:
        wav_file.write(wav_bytes)
    print(f"WAV file has been saved to {output_path}")

def get_params(req, args):
    output_dir = Path(args.output_dir)
    params = {
        "text": req.args.get("text", "").strip() or req.form.get("text", "").strip(),
        "lang": req.args.get("lang", "").strip().lower() or req.form.get("lang", "").strip().lower(),
        "role": req.args.get("role", "中文女").strip() or req.form.get("role", "中文女"),
        "reference_audio": req.args.get("reference_audio") or req.form.get("reference_audio"),
        "reference_text": req.args.get("reference_text", "").strip() or req.form.get("reference_text", ""),
        "speed": float(req.args.get("speed") or req.form.get("speed") or 1.0),
        "seed": int(req.args.get("seed") or req.form.get("seed") or -1)
    }
    if params['lang'] == 'ja': params['lang'] = 'jp'
    elif params['lang'].startswith('zh'): params['lang'] = 'zh'

    if req.args.get('encode', '') == 'base64' or req.form.get('encode', '') == 'base64':
        if params["reference_audio"]:
            tmp_name = f'{time.time()}-clone-{len(params["reference_audio"])}.wav'
            output_path = output_dir / tmp_name
            base64_to_wav(params['reference_audio'], output_path)
            params['reference_audio'] = str(output_path)
    return params

def validate_and_fix_role(role, model_type='sft'):
    """验证并修正音色参数"""
    global sft_model, tts_model
    
    # 根据模型类型选择对应的模型
    model = sft_model if model_type == 'sft' else tts_model
    
    if model is None:
        # 如果模型未加载，返回原始role，让后续加载时处理
        return role, None
    
    try:
        available_spks = model.list_available_spks()
        if role in available_spks:
            return role, None
        else:
            # 如果指定的音色不存在，使用默认音色
            default_role = available_spks[0] if available_spks else "中文女"
            warning_msg = f"指定的音色 '{role}' 不存在，已自动使用默认音色 '{default_role}'"
            app.logger.warning(warning_msg)
            return default_role, warning_msg
    except Exception as e:
        app.logger.error(f"验证音色时出错: {e}")
        return role, None

def batch(tts_type, outname, params, args):
    from cosyvoice.utils.file_utils import load_wav

    # Seed priority: API param > command-line arg > random
    seed = args.seed  # Start with global seed as a fallback
    api_seed = params.get('seed', -1)
    if api_seed != -1:
        seed = api_seed  # API-level seed takes precedence

    # If no seed was provided by API or command line, generate a random one
    if seed == -1:
        seed = random.randint(1, 100000000)

    print(f"Using seed: {seed}")
    # 关键修复：只在用户明确指定seed时才设置，否则保持WebUI的自然行为
    if api_seed != -1:
        # 用户明确指定了seed，才设置随机种子
        set_all_random_seed(seed)
        print(f"User specified seed: {seed}")
    else:
        # 用户未指定seed，不设置随机种子，让模型保持自然的差异化
        print(f"No seed specified, using natural voice differentiation")

    output_dir = Path(args.output_dir)
    reference_dir = Path(args.refer_audio_dir)

    if tts_type == 'tts':
        load_model('sft', args)
    else:
        load_model('tts', args)

    model = sft_model if tts_type == 'tts' else tts_model

    prompt_speech_16k = None
    if tts_type != 'tts':
        ref_audio_path_str = params['reference_audio']
        if not ref_audio_path_str:
            raise Exception('参考音频未传入。')

        # FIX: Clearer variable names to avoid confusion
        user_provided_path = Path(ref_audio_path_str)
        full_ref_path = user_provided_path
        if not user_provided_path.is_absolute():
            full_ref_path = reference_dir / user_provided_path

        if not full_ref_path.exists():
            raise Exception(f'参考音频不存在: {full_ref_path}')

        # Align with webui.py by removing the explicit ffmpeg call.
        # The load_wav function is expected to handle resampling.
        # Also, use model.sample_rate for postprocessing padding to match webui.py.
        prompt_speech_16k = postprocess(load_wav(str(full_ref_path), 16000), sample_rate=model.sample_rate)

    text = params['text']
    audio_list = []

    if tts_type == 'tts':
        # 完全按照WebUI的方式调用
        for i in model.inference_sft(text, params['role'], stream=False, speed=params['speed']):
            audio_list.append(i['tts_speech'])
    elif tts_type == 'clone_eq' and params.get('reference_text'):
        for i in model.inference_zero_shot(text, params.get('reference_text'), prompt_speech_16k, stream=False, speed=params['speed']):
            audio_list.append(i['tts_speech'])
    else:  # clone_mul
        for i in model.inference_cross_lingual(text, prompt_speech_16k, stream=False, speed=params['speed']):
            audio_list.append(i['tts_speech'])

    if not audio_list:
        raise Exception("模型未能生成任何音频数据。")

    audio_data = torch.cat(audio_list, dim=1)
    sample_rate = model.sample_rate

    output_path = output_dir / outname

    # Use torchaudio's save function with soundfile backend to avoid torchcodec dependency
    try:
        torchaudio.save(str(output_path), audio_data, sample_rate, format="wav", backend='soundfile')
    except ImportError as e:
        if 'torchcodec' in str(e):
            # Fallback to default backend if torchcodec is not available
            torchaudio.save(str(output_path), audio_data, sample_rate, format="wav")
        else:
            raise e

    print(f"音频文件生成成功：{output_path}")
    return str(output_path)

# --- Flask Routes ---

@app.route('/voices', methods=['GET'])
def list_voices():
    """返回可用音色列表"""
    try:
        # 确保SFT模型已加载
        load_model('sft', app.config['args'])
        
        if sft_model is None:
            return make_response(jsonify({"code": 3, "msg": "SFT模型未加载"}), 500)
        
        available_spks = sft_model.list_available_spks()
        
        return jsonify({
            "code": 0,
            "msg": "success",
            "data": {
                "voices": available_spks,
                "count": len(available_spks),
                "default_voice": available_spks[0] if available_spks else "中文女"
            }
        })
    except Exception as e:
        app.logger.error(f"List Voices Error: {e}", exc_info=True)
        return make_response(jsonify({"code": 4, "msg": str(e)}), 500)

@app.route('/tts', methods=['GET', 'POST'])
def tts():
    try:
        params = get_params(request, app.config['args'])
        if not params['text']:
            return make_response(jsonify({"code": 1, "msg": '缺少待合成的文本'}), 400)
        
        # 验证并修正音色参数
        original_role = params['role']
        params['role'], warning_msg = validate_and_fix_role(params['role'], 'sft')
        
        outname = f"tts-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.wav"
        outfile = batch(tts_type='tts', outname=outname, params=params, args=app.config['args'])
        
        # 如果有警告信息，在响应头中添加
        response = send_file(outfile, mimetype='audio/x-wav')
        if warning_msg:
            response.headers['X-Voice-Warning'] = warning_msg
            response.headers['X-Original-Voice'] = original_role
            response.headers['X-Used-Voice'] = params['role']
        
        return response
    except Exception as e:
        app.logger.error(f"TTS Error: {e}", exc_info=True)
        return make_response(jsonify({"code": 2, "msg": str(e)}), 500)

@app.route('/clone_mul', methods=['GET', 'POST'])
@app.route('/clone', methods=['GET', 'POST'])
def clone():
    try:
        params = get_params(request, app.config['args'])
        if not params['text']:
            return make_response(jsonify({"code": 6, "msg": '缺少待合成的文本'}), 400)
        outname = f"clone-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.wav"
        outfile = batch(tts_type='clone_mul', outname=outname, params=params, args=app.config['args'])
        return send_file(outfile, mimetype='audio/x-wav')
    except Exception as e:
        app.logger.error(f"Clone Error: {e}", exc_info=True)
        return make_response(jsonify({"code": 8, "msg": str(e)}), 500)

@app.route('/clone_eq', methods=['GET', 'POST'])
def clone_eq():
    try:
        params = get_params(request, app.config['args'])
        if not params['text']:
            return make_response(jsonify({"code": 6, "msg": '缺少待合成的文本'}), 400)
        if not params['reference_text']:
            return make_response(jsonify({"code": 7, "msg": '同语言克隆必须传递引用文本'}), 400)
        outname = f"clone_eq-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.wav"
        outfile = batch(tts_type='clone_eq', outname=outname, params=params, args=app.config['args'])
        return send_file(outfile, mimetype='audio/x-wav')
    except Exception as e:
        app.logger.error(f"Clone EQ Error: {e}", exc_info=True)
        return make_response(jsonify({"code": 8, "msg": str(e)}), 500)

@app.route('/v1/audio/speech', methods=['POST'])
def audio_speech():
    import random
    if not request.is_json: return jsonify({"error": "请求必须是 JSON 格式"}), 400
    data = request.get_json()
    if 'input' not in data or 'voice' not in data: return jsonify({"error": "请求缺少必要的参数： input, voice"}), 400

    params = {
        'text': data.get('input'),
        'speed': float(data.get('speed', 1.0)),
        'role': data.get('voice', '中文女'),
        'reference_audio': None
    }

    api_name = 'tts'
    if params['role'] not in VOICE_LIST:
        api_name = 'clone_mul'
        params['reference_audio'] = params['role']

    filename = f'openai-{len(params["text"] )}-{time.time()}-{random.randint(1000,99999)}.wav'
    try:
        outfile = batch(tts_type=api_name, outname=filename, params=params, args=app.config['args'])
        return send_file(outfile, mimetype='audio/x-wav')
    except Exception as e:
        app.logger.error(f"OpenAI API Error: {e}", exc_info=True)
        return jsonify({"error": {"message": str(e), "type": e.__class__.__name__}}), 500

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CosyVoice API Server", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--port', type=int, default=9233, help='Port to bind the server to.')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to.')
    parser.add_argument('--models-dir', type=str, default='./pretrained_models', help='Directory to store and load models from.')
    parser.add_argument('--sft-model-path', type=str, default=None, help='Custom path for SFT model (overrides default model search).')
    parser.add_argument('--tts-model-path', type=str, default=None, help='Custom path for TTS model (overrides default model search).')
    parser.add_argument('--output-dir', type=str, default='./tmp', help='Directory to save generated audio files.')
    parser.add_argument('--refer-audio-dir', type=str, default='.', dest='refer_audio_dir', help='Base directory for reference audio files.')
    parser.add_argument('--seed', type=int, default=-1, help='Global random seed. -1 for random. Overridden by seed in API call.')
    parser.add_argument('--preload-models', nargs='*', choices=['sft', 'tts'], default=[], help='Space-separated list of models to preload at startup (e.g., sft tts).')
    parser.add_argument('--disable-download', action='store_true', help='Disable automatic model downloading.')
    args = parser.parse_args()

    app.config['args'] = args

    output_dir = Path(args.output_dir)
    logs_dir = output_dir / 'logs'
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    app.static_folder = str(output_dir)
    app.static_url_path = '/' + output_dir.name

    setup_logging(logs_dir)
    setup_environment()

    for model_key in args.preload_models:
        try:
            load_model(model_key, args)
        except Exception as e:
            app.logger.error(f"Failed to preload model '{model_key}': {e}", exc_info=True)
            sys.exit(1)

    print(f"\n--- CosyVoice API Server ---")
    print(f"- Host: {args.host}")
    print(f"- Port: {args.port}")
    print(f"- Models Dir: {Path(args.models_dir).resolve()}")
    print(f"- Output Dir: {Path(args.output_dir).resolve()}")
    print(f"- Reference Dir: {Path(args.refer_audio_dir).resolve()}")
    print(f"- Preloaded models: {args.preload_models if args.preload_models else 'None'}")
    print(f"- Auto-download: {'Disabled' if args.disable_download else 'Enabled'}")
    print(f"- API running at: http://{args.host}:{args.port}")
    print(f"----------------------------")

    try:
        from waitress import serve
        serve(app, host=args.host, port=args.port)
    except ImportError:
        app.run(host=args.host, port=args.port)
