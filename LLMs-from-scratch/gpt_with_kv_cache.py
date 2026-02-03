import torch
import torch.nn as nn
import tiktoken


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

    def forward(self, x, use_cache=False):
        shortcut = x
        x = self.norm1(x)
        # 增加缓存
        x = self.att(x, use_cache=use_cache)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class MultiHeadAttention(torch.nn.Module):
    """
    思路：先投影、后切分、并行计算、再拼接、最后融合。
    把高维的 Q/K/V 向量，切分成num_heads个低维的子向量，
    让每个子向量独立计算注意力（学习不同的语义关联），
    最后把结果合并，实现「用一套参数完成多头计算」的高效目标。
    核心维度变化：
    输入x: [b, n, d_in]
    → Q/K/V投影: [b, n, D]
    → view切分头: [b, n, h, d]
    → transpose调整顺序: [b, h, n, d]
    → 注意力分数: [b, h, n, n]
    → 权重加权求和V: [b, h, n, d]
    → transpose恢复顺序: [b, n, h, d]
    → view拼接多头: [b, n, D]
    → out_proj融合: [b, n, D]
    输出context_vec: [b, n, D]

    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 强制校验：输出维度必须能被注意力头数整除，否则无法均分
        # 我们要把d_out这个总维度，平均切分给num_heads个注意力头，
        # 每个头分到的维度就是head_dim = d_out / num_heads，
        # 如果不能整除，就会出现维度残缺，程序直接报错。
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        # 张量重塑 / 转置的核心依据，必须定义为实例变量
        self.d_out = d_out  # 多头注意力的最终总输出维度
        self.num_heads = num_heads  # 多头注意力的最终总输出维度
        # 黄金公式：d_out = num_heads × head_dim
        # Transformer 多头注意力的核心维度公式，所有大模型都遵循这个公式
        self.head_dim = d_out // num_heads  # 每个注意力头的维度(核心！均分)

        # 核心：只用1套Q/K/V线性层，投影到总输出维度d_out 参数量骤减，计算效率翻倍
        # 先把所有头的特征一次性投影出来，再切分，而不是每个头单独投影，本质是「批量处理」
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        # 组合多头的输出 多头输出的融合投影层
        self.out_proj = torch.nn.Linear(d_out, d_out)
        # 对注意力权重做随机失活，防止过拟合
        self.dropout = torch.nn.Dropout(dropout)
        # 创建一个上三角的屏蔽矩阵，用于屏蔽掉 Future Tokens
        self.register_buffer("mask",
                             torch.triu(torch.ones(context_length, context_length), diagonal=1))
        # 添加了两个缓冲区 cache_k 和 cache_v
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def forward(self, x, use_cache=False):
        # 输入shape: [批次, 序列长度, 输入特征维度]
        b, num_tokens, d_in = x.shape
        # 步骤1：通过线性层做一次整体投影，得到高维的Q/K/V
        keys = self.W_key(x)  # shape: [b, num_tokens, d_out]
        queries = self.W_query(x)
        values = self.W_value(x)
        # 步骤2：重塑张量，切分成多个注意力头 → 均分特征维度
        # 核心意义：把原本的高维特征，切分成h个独立的低维子特征，每个子特征对应一个注意力头，每个头只处理自己的d维特征。
        # d_out = num_heads * head_dim
        # shape变化：[b, num_tokens, d_out] → [b, num_tokens, num_heads, head_dim]
        keys_new = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 检索缓存
        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
            keys, values = self.cache_k, self.cache_v
        else:
            keys, values = keys_new, values_new

        # 步骤3：转置调整维度顺序，把「头维度」提到前面，方便并行计算 让 PyTorch 可以对h个注意力头做并行计算！
        # shape变化：[b, num_tokens, num_heads, head_dim] → [b, num_heads, num_tokens, head_dim]
        # 调整前：批次→序列→头→特征 → 头是第三维，计算时是串行；
        # 调整后：批次→头→序列→特征 → 头是第二维，所有头可以同时计算注意力，GPU 的并行算力被完全利用，速度提升 h 倍！
        keys = keys.transpose(1, 2)  # 交换张量的第 1 维和第 2 维
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 步骤4：计算注意力分数（核心矩阵乘法）
        # shape 变化：[b,h,n,d] × [b,h,d,n] = [b,h,n,n]
        # 核心意义：attn_scores[b, h, i, j]
        # 表示「第 b 个样本、第 h 个注意力头、
        # 第 i 个 token 对 第 j 个 token 的注意力匹配度」，值越大，越关注。
        attn_scores = queries @ keys.transpose(2, 3)  # 把 keys 的[b, num_heads, num_tokens, head_dim] →[b, num_heads ,head_dim, num_tokens]

        # 步骤5：因果掩码，屏蔽未来token的注意力分数
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 步骤6：分数缩放 + softmax归一化 → 注意力权重
        # 对最后一维（序列维度）做归一化，权重之和为 1，得到注意力权重
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)  # 随机失活防止过拟合

        # 步骤7：权重加权求和V + 恢复维度顺序
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 步骤8：拼接所有头的输出，恢复成完整维度
        # torch.contiguous()连续化张量
        # 作用：因为前面做了多次transpose转置，张量的内存布局会变成「非连续」，此时直接调用view会报错；
        # contiguous()：重新整理张量的内存布局，变成连续的，不改变张量的值，只为了后续view能正常执行。
        # torch.view() 核心意义：把h个独立的头特征，在特征维度上拼接成一个完整的特征向量，
        # 恢复到总输出维度D，完成了「多头特征的拼接」。
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # 步骤9：多头输出融合投影
        context_vec = self.out_proj(context_vec)
        return context_vec

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 位置嵌入层：为模型注入位置信息
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # 嵌入层 Dropout：防止过拟合
        # nn.ModuleList：是实现自定义执行逻辑的唯一选择，比如大模型中的「动态层选择」「MoE（混合专家模型）的层调度」「推理时的层裁剪」等，都需要通过nn.ModuleList手动控制；
        # 模型的核心能力层，通过多层Transformer块的堆叠，对嵌入向量进行层层特征提取，捕捉文本中的长距离依赖、语法结构、语义关联等信息
        # 在forward方法 必须手动循环，显式传递数据
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg)
             for _ in range(cfg["n_layers"])]
        )
        self.current_pos = 0
        # 最终归一化层：稳定输出特征分布
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 输出头：将特征向量映射为词汇表概率
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx, use_cache=False):
        batch_size, seq_len = in_idx.shape
        # 计算词嵌入向量
        tok_embeds = self.tok_emb(in_idx)
        # 计算位置嵌入向量
        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        ### 缓存
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        # 融合词嵌入和位置嵌入
        # 核心逻辑：将 语义信息（词嵌入）和位置信息（位置嵌入）融合，让模型同时理解 “token的含义” 和 “token在序列中的位置”。
        x = tok_embeds + pos_embeds
        # 嵌入层 Dropout
        x = self.drop_emb(x)
        # 通过 Transformer 块堆叠提取特征
        # x = self.trf_blocks(x)
        # nn.ModuleList 必须手动循环，显式传递数据
        for blk in self.trf_blocks:
            x = blk(x, use_cache)
        # 最终归一化
        x = self.final_norm(x)
        # 输出头计算 logits
        logits = self.out_head(x)
        return logits

    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]  # 取最后一个维度的概率
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat([idx, idx_next], dim=-1)
    return idx


def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    """
        基于自回归的LLM文本生成函数，支持KV Cache缓存优化（贪心采样）
        :param model: 训练完成的LLM模型（需实现KV Cache接口）
        :param idx: 初始token索引张量，shape [batch_size, seq_len]
        :param max_new_tokens: 最大生成新token数量
        :param context_size: 最大上下文窗口长度，None则使用模型默认值
        :param use_cache: 是否启用KV Cache加速
        :return: 原始+生成的完整token索引张量
        """
    # 1. 将模型设为评估模式，禁用训练相关层（Dropout/BatchNorm等），保证推理一致性
    model.eval()
    # 2. 确定模型推理的最大上下文窗口长度：优先用传入的context_size，否则用模型位置嵌入层的最大长度
    # model.pos_emb.num_embeddings → 位置嵌入层的嵌入数量，即模型预定义的最大上下文长度
    ctx_len = context_size or model.pos_emb.num_embeddings
    # 3. 禁用梯度计算：推理阶段无需反向传播，大幅减少显存占用、提升推理速度
    with torch.no_grad():
        if use_cache:
            # 3.1 重置模型的KV Cache缓存区：生成新序列前清空历史缓存，避免跨任务污染
            model.reset_kv_cache()
            # 3.2 首次前向传播：输入取最后ctx_len个token（防止超出上下文窗口），开启cache
            # 作用：计算初始序列的logits，并将历史token的K/V存入缓存区
            logits = model(idx[:, -ctx_len:], use_cache=True)
            # 3.3 自回归循环生成max_new_tokens个新token
            for _ in range(max_new_tokens):
                # 3.4 贪心采样：取最后一个位置的logits，按最后一维取argmax（得分最高的token）
                # keepdim=True → 保持维度为[batch_size, 1]，避免维度坍塌，方便后续拼接
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # 3.5 拼接新token：将生成的next_idx拼接到原始idx后，更新输入序列
                idx = torch.cat([idx, next_idx], dim=1)
                # 3.6 后续前向传播：仅输入新生成的单个token，开启cache
                # 核心优化：模型从缓存中读取历史K/V，仅计算新token的注意力，无需重复计算历史
                logits = model(next_idx, use_cache=True)
        else:
            # 3.7 自回归循环生成max_new_tokens个新token
            for _ in range(max_new_tokens):
                # 3.8 每次前向传播都输入最后ctx_len个token，禁用cache
                # 缺点：重复计算所有历史token的注意力，生成越长，速度越慢
                logits = model(idx[:, -ctx_len:], use_cache=False)
                # 3.9 贪心采样获取下一个token
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # 3.10 同分支1，拼接新token更新输入序列
                idx = torch.cat([idx, next_idx], dim=1)

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
    out = generate_text_simple(model,
                               encoded_tensor,
                               max_new_tokens=6,
                               context_size=GPT_CONFIG_124M["context_length"])
    print("Output:", out)
    print("Output length:", len(out[0]))

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)


if __name__ == "__main__":
    main()
