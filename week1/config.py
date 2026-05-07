import torch
import transformers
import cv2

print("=== XrayVision 환경 확인 ===")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Transformers: {transformers.__version__}")
print(f"OpenCV: {cv2.__version__}")
print("환경 세팅 완료")
