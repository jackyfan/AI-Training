import torch
import torch.nn as nn


def step_by_by_step_attention():
    # ===================== 第一步：文本预处理 + 构建词汇表映射 =====================
    # 原始文本：待处理的英文句子
    from contextlib import suppress

    sentence = 'Life is short, eat dessert first'

    # 核心操作：构建"单词→索引"的字典映射（Tokenizer的极简实现）
    # 拆解步骤：
    # 1. sentence.replace(',', '')：去掉句子中的逗号，得到 'Life is short eat dessert first'
    # 2. .split()：按空格分割成单词列表 → ['Life', 'is', 'short', 'eat', 'dessert', 'first']
    # 3. sorted()：对单词列表按字母顺序排序 → ['Life', 'dessert', 'eat', 'first', 'is', 'short']
    # 4. enumerate()：遍历排序后的列表，生成(索引, 单词)对 → (0,'Life'), (1,'dessert'), ..., (5,'short')
    # 5. {s:i for i,s in ...}：字典推导式，最终得到"单词:索引"的映射（key是单词，value是索引）
    dc = {s: i for i, s in enumerate(sorted(sentence.replace(',', '').split()))}
    print(dc)  # 输出：{'Life':0, 'dessert':1, 'eat':2, 'first':3, 'is':4, 'short':5}

    # ===================== 第二步：文本→数字编码（Tokenization） =====================
    import torch  # 导入PyTorch库

    # 核心操作：将原始句子的每个单词转换为对应的数字索引（大模型的输入必须是数字）
    # 拆解步骤：
    # 1. sentence.replace(',', '').split()：还原成原始单词列表 → ['Life', 'is', 'short', 'eat', 'dessert', 'first']
    # 2. [dc[s] for s in ...]：遍历单词列表，用dc字典查每个单词的索引 → [0,4,5,2,1,3]
    # 3. torch.tensor(...)：将列表转为PyTorch张量（大模型的标准输入格式）
    sentence_int = torch.tensor([dc[s] for s in sentence.replace(',', '').split()])
    print(sentence_int)  # 输出：tensor([0, 4, 5, 2, 1, 3]) → 每个单词对应的数字索引

    # ===================== 第三步：词嵌入（将数字索引→连续向量） =====================
    # 词汇表大小：这里实际只用了6个单词，设为50_000是模拟真实大模型的词汇表规模
    # 50_000中的下划线是Python数字分隔符，仅提升可读性，等价于50000
    vocab_size = 50_000

    # 设置随机种子：固定所有随机操作的结果，保证实验可复现（谷歌工程方法论核心）c
    torch.manual_seed(123)

    # 创建词嵌入层：nn.Embedding(词汇表大小, 嵌入维度)
    # 作用：将离散的数字索引映射为可学习的连续向量，这里嵌入维度设为3（简化版，真实大模型是数百/数千维）
    embed = torch.nn.Embedding(vocab_size, 3)

    # 对数字编码的句子做嵌入，.detach()：剥离计算图，仅获取数值（避免后续计算影响梯度）
    embedded_sentence = embed(sentence_int).detach()
    print(embedded_sentence)  # 输出6个3维向量（对应6个单词）
    print(embedded_sentence.shape)  # 输出：torch.Size([6, 3]) → [token数量, 嵌入维度]

    # ===================== 第四步：定义Query/Key/Value投影矩阵（注意力核心） =====================
    # 重置随机种子：确保投影矩阵初始化结果固定
    torch.manual_seed(123)

    # d：嵌入维度（这里d=3，即embedded_sentence.shape[1]）
    d = embedded_sentence.shape[1]

    # 定义Query/Key/Value的维度（简化版，真实多头注意力中d_q=d_k=d_v=head_dim）
    d_q, d_k, d_v = 2, 2, 4

    # 定义可学习的投影矩阵（nn.Parameter：标记为模型可训练参数）
    # torch.rand(d, d_q)：生成d×d_q的随机矩阵（这里是3×2），对应Query投影
    W_query = torch.nn.Parameter(torch.rand(d, d_q))
    W_key = torch.nn.Parameter(torch.rand(d, d_k))  # Key投影矩阵（3×2）
    W_value = torch.nn.Parameter(torch.rand(d, d_v))  # Value投影矩阵（3×4）

    print(W_query.shape)  # 输出：torch.Size([3, 2])
    print(W_key.shape)  # 输出：torch.Size([3, 2])
    print(W_value.shape)  # 输出：torch.Size([3, 4])

    # ===================== 第五步：单个token的Q/K/V计算（以第2个token为例） =====================
    # x_2：取embedded_sentence中索引为1的向量（对应单词"is"，Python索引从0开始）
    x_2 = embedded_sentence[1]
    print(x_2.shape)  # 输出：torch.Size([3]) → 单个token的嵌入向量

    # Query投影：x_2（3维） @ W_query（3×2） → 得到2维Query向量
    query_2 = x_2 @ W_query
    print(query_2.shape)  # 输出：torch.Size([2])

    # Key投影：x_2（3维） @ W_key（3×2） → 得到2维Key向量
    key_2 = x_2 @ W_key
    print(key_2.shape)  # 输出：torch.Size([2])

    # Value投影：x_2（3维） @ W_value（3×4） → 得到4维Value向量
    value_2 = x_2 @ W_value
    print(value_2.shape)  # 输出：torch.Size([4])

    # ===================== 第六步：所有token的Key/Value投影 =====================
    # 对整个句子的嵌入向量做Key投影：embedded_sentence(6×3) @ W_key(3×2) → 6×2
    keys = embedded_sentence @ W_key
    # 对整个句子的嵌入向量做Value投影：embedded_sentence(6×3) @ W_value(3×4) → 6×4
    values = embedded_sentence @ W_value

    print("embedded_sentence.shape:", embedded_sentence.shape)  # torch.Size([6, 3])
    print("keys.shape:", keys.shape)  # torch.Size([6, 2])
    print("values.shape:", values.shape)  # torch.Size([6, 4])

    # ===================== 第七步：注意力分数计算（核心：衡量token间的关联度） =====================
    # omega_24：计算第2个token的Query与第5个token的Key的点积（注意力分数）
    # query_2(2维) .dot(keys[4](2维)) → 标量值，代表两个token的关联程度
    omega_24 = query_2.dot(keys[4])
    print(omega_24)  # 输出一个标量（比如：0.5678）

    # omega_2：计算第2个token与所有token的注意力分数
    # query_2(2维) @ keys.T(2×6) → 6维向量，每个值对应与一个token的关联度
    # keys.T：对keys矩阵转置（6×2 → 2×6）
    omega_2 = query_2 @ keys.T
    print(omega_2)  # 输出：tensor([x1, x2, x3, x4, x5, x6]) → 6个注意力分数

    # ===================== 第八步：注意力权重归一化（Softmax） =====================
    import torch.nn.functional as F

    # 核心操作：计算注意力权重（Softmax归一化，除以√d_k是为了防止分数过大）
    # 1. omega_2 / d_k**0.5：缩放注意力分数（d_k=2，√2≈1.414）
    # 2. F.softmax(..., dim=0)：在第0维做softmax，使所有权重和为1
    attention_weights_2 = F.softmax(omega_2 / d_k ** 0.5, dim=0)
    print(attention_weights_2)  # 输出6个权重值，总和=1（比如：[0.1, 0.2, 0.15, 0.25, 0.1, 0.2]）

    # ===================== 第九步：计算上下文向量（注意力加权求和） =====================
    # 核心操作：用注意力权重对所有token的Value向量加权求和，得到最终上下文向量
    # attention_weights_2(6维) @ values(6×4) → 4维上下文向量
    context_vector_2 = attention_weights_2 @ values
    print(context_vector_2.shape)  # 输出：torch.Size([4])
    print(context_vector_2)  # 输出4维上下文向量（第2个token的最终注意力输出）


class SelfAttention(nn.Module):
    r"""
        Args:
            d_in:输入特征向量的维度
            d_out_kq:查询和键输出的维度
            d_out_v:值输出的维度
        """

    def __init__(self, d_in, d_out_kq, d_out_v):
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = torch.nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key = torch.nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_value = torch.nn.Parameter(torch.rand(d_in, d_out_v))

    def forward(self, x):
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / self.d_out_kq ** 0.5, dim=-1
        )
        context_vector = attn_weights @ values
        return context_vector


class MultiHeadAttentionWrapper(nn.Module):
    r"""
    Args:
        d_in:输入特征向量的维度
        d_out_kq:查询和键输出的维度
        d_out_v:值输出的维度
        num_heads:注意力头的数量
    """

    def __init__(self, d_in, d_out_kq, d_out_v, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([
            SelfAttention(d_in, d_out_kq, d_out_v)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


if __name__ == "__main__":
    sentence = 'Life is short, eat dessert first'
    dc = {s: i for i, s in enumerate(sorted(sentence.replace(',', '').split()))}
    sentence_int = torch.tensor([dc[s] for s in sentence.replace(',', '').split()])
    vocab_size = 50_000
    torch.manual_seed(123)
    embed = torch.nn.Embedding(vocab_size, 3)
    embedded_sentence = embed(sentence_int).detach()
    torch.manual_seed(123)
    d_in, d_out_kq, d_out_v = 3, 2, 1
    sa = SelfAttention(d_in, d_out_kq, d_out_v)
    print(sa(embedded_sentence))
    torch.manual_seed(123)
    block_size = embedded_sentence.shape[1]
    mha = MultiHeadAttentionWrapper(
        d_in, d_out_kq, d_out_v, num_heads=4
    )
    context_vecs = mha(embedded_sentence)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)
