# CosyVoice 性能优化指南

本文档介绍了 CosyVoice 的性能优化方案，包括 FP16 加速、TensorRT 优化、智能文本分割等功能。

## 🚀 快速开始

### 方法1: 使用优化启动器（推荐）
```bash
# 自动检测硬件并应用最佳配置
python start_optimized_server.py

# 指定端口
python start_optimized_server.py --port 9234

# 启动前预编译TensorRT引擎
python start_optimized_server.py --precompile

# 启动前运行性能测试
python start_optimized_server.py --benchmark
```

### 方法2: 直接启动优化服务器
```bash
# 启用所有优化（推荐配置）
python fastapi_server.py --model_dir pretrained_models/CosyVoice-300M-SFT --port 9234 --load_trt --load_jit --fp16

# 仅启用FP16和JIT（适合低显存）
python fastapi_server.py --model_dir pretrained_models/CosyVoice-300M-SFT --port 9234 --load_jit --fp16

# 高并发配置（8GB+显存）
python fastapi_server.py --model_dir pretrained_models/CosyVoice-300M-SFT --port 9234 --load_trt --load_jit --fp16 --trt_concurrent 2
```

## 📊 性能优化方案

### 1. FP16 半精度加速
- **提升**: 30-50% 速度提升
- **内存**: 减少约50%显存占用
- **启用**: `--fp16`
- **适用**: 支持FP16的现代GPU

### 2. TensorRT 加速
- **提升**: 显著提升推理速度
- **特点**: 首次启动需要编译时间
- **启用**: `--load_trt`
- **优化**: 使用预编译脚本避免每次编译

### 3. JIT 编译加速
- **提升**: 加速LLM部分推理
- **特点**: 启动时编译，后续推理更快
- **启用**: `--load_jit`
- **兼容**: 与TensorRT配合使用效果更佳

### 4. 智能文本分割
- **功能**: 自动分割长文本，优化处理速度
- **参数**: `max_text_length`（默认100字符）
- **启用**: `enable_smart_split=True`（默认启用）

### 5. 流式合成
- **功能**: 支持流式音频输出
- **参数**: `stream=True`
- **优势**: 降低首字延迟，提升用户体验

## 🛠️ 工具脚本

### TensorRT 预编译
```bash
# 预编译TensorRT引擎，避免每次启动编译
python precompile_trt.py --model_dir pretrained_models/CosyVoice-300M-SFT --fp16

# 指定并发数
python precompile_trt.py --model_dir pretrained_models/CosyVoice-300M-SFT --fp16 --trt_concurrent 2
```

### 性能基准测试
```bash
# 测试当前配置性能
python benchmark_performance.py --model_dir pretrained_models/CosyVoice-300M-SFT --load_trt --load_jit --fp16

# 测试所有配置组合
python benchmark_performance.py --model_dir pretrained_models/CosyVoice-300M-SFT --test_all
```

## 📈 性能等级说明

| RTF 范围 | 性能等级 | 说明 |
|---------|---------|------|
| < 0.3 | 🚀 非常快 | 实时性极佳，适合实时对话 |
| 0.3-0.7 | ⚡ 快速 | 适合实时应用 |
| 0.7-1.0 | ✅ 可接受 | 接近实时，一般应用可用 |
| > 1.0 | 🐌 较慢 | 需要优化 |

## 🔧 API 端点

### 基础合成
```bash
POST /inference_sft
```
参数：
- `tts_text`: 要合成的文本
- `spk_id`: 音色ID
- `format`: 输出格式（stream/wav）
- `speed`: 语速（默认1.0）
- `stream`: 是否流式合成
- `max_text_length`: 文本分割长度
- `enable_smart_split`: 是否启用智能分割

### 管理端点
- `GET /list_voices` - 获取音色列表
- `POST /warmup_trt` - 预热TensorRT引擎
- `GET /performance_info` - 获取性能配置信息
- `GET /docs` - API文档

## 💡 优化建议

### 硬件配置推荐
- **8GB+ 显存**: 启用所有优化 + 并发数2
- **4-8GB 显存**: 启用TRT + JIT + FP16
- **2-4GB 显存**: 启用JIT + FP16（不启用TRT）
- **<2GB 显存**: 仅启用JIT

### 使用技巧
1. **首次启动**: 使用预编译脚本生成TensorRT引擎
2. **长文本**: 启用智能分割，设置合适的分割长度
3. **实时应用**: 使用流式合成降低延迟
4. **批量处理**: 适当增加TRT并发数

### 故障排除
1. **TensorRT编译失败**: 检查CUDA版本兼容性
2. **显存不足**: 降低并发数或禁用TensorRT
3. **速度仍然慢**: 运行基准测试对比不同配置

## 📋 配置示例

### 高性能配置（8GB+显存）
```bash
python fastapi_server.py \
  --model_dir pretrained_models/CosyVoice-300M-SFT \
  --port 9234 \
  --load_trt \
  --load_jit \
  --fp16 \
  --trt_concurrent 2
```

### 标准配置（4-8GB显存）
```bash
python fastapi_server.py \
  --model_dir pretrained_models/CosyVoice-300M-SFT \
  --port 9234 \
  --load_trt \
  --load_jit \
  --fp16
```

### 轻量配置（2-4GB显存）
```bash
python fastapi_server.py \
  --model_dir pretrained_models/CosyVoice-300M-SFT \
  --port 9234 \
  --load_jit \
  --fp16
```

## 🔍 监控和调试

### 性能监控
- 查看控制台输出的RTF值
- 使用 `/performance_info` 端点检查配置
- HTTP响应头包含处理时间信息

### 调试信息
- 长文本分割日志
- TensorRT编译状态
- 模型加载信息

## 🚨 故障排除

### TensorRT 初始化问题

如果遇到类似错误：
```
[TRT] [W] Unable to determine GPU memory usage
[TRT] [I] [MemUsageChange] Init CUDA: CPU +0, GPU +0, now: CPU 62019, GPU 0 (MiB)
```

**解决方案**:

1. **运行诊断脚本**:
   ```bash
   python diagnose_gpu.py
   ```

2. **使用备用启动器**:
   ```bash
   # 自动检测可用配置
   python start_fallback_server.py --port 9234
   
   # 强制使用安全配置
   python start_fallback_server.py --force_config jit_fp16 --port 9234
   ```

3. **手动降级配置**:
   ```bash
   # 如果TensorRT有问题，使用JIT+FP16
   python fastapi_server.py --model_dir pretrained_models/CosyVoice-300M-SFT --port 9234 --load_jit --fp16
   
   # 如果仍有问题，仅使用JIT
   python fastapi_server.py --model_dir pretrained_models/CosyVoice-300M-SFT --port 9234 --load_jit
   ```

### 常见问题

| 问题 | 症状 | 解决方案 |
|------|------|----------|
| CUDA不可用 | `CUDA 不可用` | 检查GPU驱动，重装PyTorch GPU版 |
| ONNXRuntime问题 | `CUDAExecutionProvider not available` | `pip install onnxruntime-gpu` |
| TensorRT编译失败 | TensorRT初始化卡住 | 使用非TRT配置或更新TensorRT |
| 显存不足 | OOM错误 | 降低并发数或禁用FP16 |
| 模型加载慢 | 启动时间长 | 使用预编译脚本 |

### 工具脚本

| 脚本 | 用途 | 命令 |
|------|------|------|
| `diagnose_gpu.py` | 环境诊断 | `python diagnose_gpu.py` |
| `start_fallback_server.py` | 自动降级启动 | `python start_fallback_server.py` |
| `benchmark_performance.py` | 性能测试 | `python benchmark_performance.py --test_all` |
| `precompile_trt.py` | TensorRT预编译 | `python precompile_trt.py` |

## 📞 技术支持

如果遇到问题：
1. **首先运行**: `python diagnose_gpu.py`
2. **尝试备用启动**: `python start_fallback_server.py`
3. **查看日志**: 检查控制台错误信息
4. **降级配置**: 从最基础配置开始测试

### 推荐排查顺序
1. 运行诊断脚本确认环境
2. 使用备用启动器自动选择配置
3. 手动尝试不同优化级别
4. 检查模型文件完整性

---

**注意**: 首次启动TensorRT时需要编译时间，建议使用预编译脚本优化启动速度。
