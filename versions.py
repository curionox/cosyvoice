import torch
import numpy as np

print("==== PyTorch & CUDA 信息 ====")
print("PyTorch 版本:", torch.__version__)
print("是否可用 CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA (PyTorch 编译版本):", torch.version.cuda)
    print("cuDNN 版本:", torch.backends.cudnn.version())
    print("GPU 名称:", torch.cuda.get_device_name(0))

print("\n==== NumPy 信息 ====")
print("NumPy 版本:", np.__version__)

print("\n==== TensorRT 信息 ====")
try:
    import tensorrt as trt
    print("TensorRT 版本:", trt.__version__)
except ImportError:
    print("TensorRT 未安装")

print("\n==== 系统 CUDA 驱动信息 ====")
import subprocess
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(result.stdout)
except FileNotFoundError:
    print("未找到 nvidia-smi，可能没装 NVIDIA 驱动或不在 PATH 中")
