import torch
import numpy

def main():
    # 检查PyTorch版本是否为2.1.0
    print("PyTorch版本：", torch.__version__)  # 输出2.1.0即正确
    # 检查是否支持CUDA（核心：RTX 3060能否被调用）
    print("是否支持CUDA：", torch.cuda.is_available())  # 必须返回True
    # 检查显卡数量、名称（确认调用的是RTX 3060）
    print("GPU数量：", torch.cuda.device_count())  # 至少1
    print("GPU名称：", torch.cuda.get_device_name(0))  # 输出NVIDIA GeForce RTX 3060
    # 测试GPU张量计算（验证算力正常）
    x = torch.tensor([1, 2, 3]).cuda()
    print("GPU张量：", x)  # 输出tensor([1,2,3], device='cuda:0')即成功
    print("NumPy版本：", numpy.__version__)  # 预期输出：1.26.x/1.25.x等1.x系列


if __name__ == "__main__":
    main()
