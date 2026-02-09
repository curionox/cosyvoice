@echo off
chcp 65001 >nul
echo ============================================================
echo CosyVoice RTX 4060 Ti 16GB 安全启动
echo ============================================================
echo.
echo 🎮 GPU: RTX 4060 Ti 16GB
echo ⚙️  配置: JIT + FP16 (避免 TensorRT 问题)
echo 🌐 端口: 9234
echo.
echo 启动中...
echo.

python fastapi_server.py --model_dir pretrained_models/CosyVoice-300M-SFT --port 9234 --load_jit --fp16

echo.
echo ============================================================
echo 服务器已停止
echo ============================================================
pause
