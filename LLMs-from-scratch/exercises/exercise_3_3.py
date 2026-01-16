
import torch
import sys
import os

# 将上级目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention import MultiHeadAttention

context_length = 1024
d_in, d_out = 768, 768
num_heads = 12

# 创建一个随机输入张量，形状为 [batch_size, num_tokens, d_in]
batch_size = 2
num_tokens = 100
x = torch.rand(batch_size, num_tokens, d_in)

# 初始化MultiHeadAttention模型
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads)

# 运行前向传播
output = mha(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(count_parameters(mha))

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Test passed!")