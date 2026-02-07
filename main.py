import torch


def main():
    # 1. 检查PyTorch版本和CUDA支持
    print(f"PyTorch版本： {torch.__version__}")
    print(f"是否支持CUDA： {torch.cuda.is_available()}")
    print(f"GPU数量： {torch.cuda.device_count()}")

    # 2. 只有CUDA可用时才获取GPU名称
    if torch.cuda.is_available():
        print("GPU名称：", torch.cuda.get_device_name(0))
        # 测试GPU张量创建
        x = torch.tensor([1, 2, 3]).to("cuda")
        print(f"张量设备： {x.device}")
    else:
        print("未检测到CUDA，使用CPU运行")


if __name__ == "__main__":
    main()