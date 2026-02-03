import torch
import torch.nn as nn
from attention import MultiHeadAttention
import tiktoken
import time


class DummyModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


# 具有5层的深度神经网络
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]),
                          GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


# Transformer块
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]) # 位置嵌入层：为模型注入位置信息
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  #嵌入层 Dropout：防止过拟合
        # nn.Sequential：仅支持固定的线性执行逻辑，无法实现任何自定义调度，适合逻辑简单、无需额外控制的场景。
        # 模型的核心能力层，通过多层Transformer块的堆叠，对嵌入向量进行层层特征提取，捕捉文本中的长距离依赖、语法结构、语义关联等信息
        # * 解包，将List分解成一个个元素，并作为参数传入Sequential
        # 比如 nn.Sequential([TransformerBlock(cfg), TransformerBlock(cfg), TransformerBlock(cfg)])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )
        # 最终归一化层：稳定输出特征分布
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 输出头：将特征向量映射为词汇表概率
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        # 计算词嵌入向量
        tok_embeds = self.tok_emb(in_idx)
        # 计算位置嵌入向量
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        # 融合词嵌入和位置嵌入
        # 核心逻辑：将 语义信息（词嵌入）和位置信息（位置嵌入）融合，让模型同时理解 “token的含义” 和 “token在序列中的位置”。
        x = tok_embeds + pos_embeds
        # 嵌入层 Dropout
        x = self.drop_emb(x)
        # 通过 Transformer 块堆叠提取特征
        x = self.trf_blocks(x)
        # 最终归一化
        x = self.final_norm(x)
        # 输出头计算 logits
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :] #取最后一个维度的概率
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat([idx, idx_next], dim=-1)
    return idx


def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # 词汇表大小
        "context_length": 1024,  # 上下文长度
        "emb_dim": 768,  # 嵌入维度
        "n_heads": 12,  # 多套注意力数量
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False  # Query-Key-Value bias
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    model.eval()  # disable dropout
    start = time.time()
    token_ids = generate_text_simple(model,
                               encoded_tensor,
                               max_new_tokens=200,
                               context_size=GPT_CONFIG_124M["context_length"])
    #print("Output:", token_ids)
    print("Output length:", len(token_ids[0]))
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())
    print(decoded_text)
    print(f"\n\n{50 * '='}\n{22 * ' '}OUT\n{50 * '='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0]) / total_time)} tokens/sec")

if __name__ == "__main__":
    main()
