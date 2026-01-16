import torch
from sympy.physics.units.systems.si import dimex

"""
自注意力Self Attention是Transformer模型中的一种机制，
为什么叫「自注意力」？
因为 Q/K/V 都来自同一个输入 x，是模型对自身序列的 token 做注意力匹配，区别于「交叉注意力」(Q 来自解码器，K/V 来自编码器)。

它通过允许一个序列中的每个位置与同一序列中的其他所有位置进行交互并权衡其重要性，来计算出更高效的输入表示。
在自注意力机制中，“自”指的是该机制通过关联单个输入序列中的不同位置来计算注意力权重的能力。
它可以评估并学习输入本身各个部分之间的关系和依赖，比如句子中的单词或图像中的像素。
上下文向量（context vector）可以被理解为一种包含了序列中所有元素信息的嵌入向量。

"""


def simple_attention():
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your (x^1)
         [0.55, 0.87, 0.66],  # journey (x^2)
         [0.57, 0.85, 0.64],  # starts (x^3)
         [0.22, 0.58, 0.33],  # with (x^4)
         [0.77, 0.25, 0.10],  # one (x^5)
         [0.05, 0.80, 0.55]]  # step (x^6)
    )
    # 一，注意力分数：通过计算输入项之间的点积，得到一个得分向量。
    query = inputs[1]
    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        # 点积本质上是将两个向量逐个元素相乘然后对乘积求和
        attn_scores_2[i] = torch.dot(x_i, query)
    print(attn_scores_2)
    # 二，注意力权重：注意力分数归一化后获得权重向量。
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
    print("Attention weights:", attn_weights_2_tmp)
    print("Sum:", attn_weights_2_tmp.sum())

    # 使用softmax函数进行归一化更为常见
    attn_weights_2_naive = softmax_naive(attn_scores_2)
    print("Attention weights:", attn_weights_2_naive)
    print("Sum:", attn_weights_2_naive.sum())
    # 使用torch.softmax
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    print("Attention weights:", attn_weights_2)
    print("Sum:", attn_weights_2.sum())

    # 三，注意力上下文：通过计算输入项的加权和获得。
    query = inputs[1]
    context_vec_2 = torch.zeros(query.shape)
    for i, x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[i] * x_i
    print(context_vec_2)

    # 以上是计算第二个输入项的注意力
    # 计算所有查询项与所有输入项进行点积
    attn_scores = torch.empty(6, 6)
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[i, j] = torch.dot(x_i, x_j)
    print(attn_scores)
    # 矩阵乘法来得到相同的结果
    attn_scores = inputs @ inputs.T
    print(attn_scores)
    # 对所有归一化
    attn_weights = torch.softmax(attn_scores, dim=-1)
    print(attn_weights)
    # 计算所有上下文向量
    all_context_vecs = attn_weights @ inputs
    print(all_context_vecs)


def softmax_naive(x):
    """
    这种简单的softmax实现（softmax_naive）在处理大输入值或小输入值时可能会遇到数值稳定性问题，比如溢出和下溢。
    """
    return torch.exp(x) / torch.exp(x).sum(dim=0)


def self_attention():
    """
    带有可训练权重的自注意力机制，被称为缩放点积注意力（scaled dot-product attention）
    引入 3 个可训练的权重矩阵 Wq、Wk和 Wv，将嵌入的输入词元 x (i)分别映射为查询向量、键向量和值向量。
    """
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your (x^1)
         [0.55, 0.87, 0.66],  # journey (x^2)
         [0.57, 0.85, 0.64],  # starts (x^3)
         [0.22, 0.58, 0.33],  # with (x^4)
         [0.77, 0.25, 0.10],  # one (x^5)
         [0.05, 0.80, 0.55]]  # step (x^6)
    )
    # 第二个输入元素
    x_2 = inputs[1]
    # 输入嵌入维度
    d_in = inputs.shape[1]
    # 输出嵌入维度
    d_out = 2
    # 初始化三个权重矩阵
    torch.manual_seed(123)
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    print(W_query)
    # 一、计算查询向量、键向量和值向量
    # 查询向量query_2是通过第二个输入元素x_2与查询权重矩阵W_query之间的矩阵乘法得到的
    query_2 = x_2 @ W_query
    key_2 = x_2 @ W_key
    value_2 = x_2 @ W_value
    print(query_2)

    keys = inputs @ W_key
    values = inputs @ W_value
    print("keys.shape:", keys.shape)
    print("values.shape:", values.shape)
    # 二、计算注意力分数w^22：查询向量query_2与键向量keys之间的点积
    keys_2 = keys[1]
    attn_score_22 = query_2.dot(keys_2)
    print(attn_score_22)
    # 计算所有注意力分数
    attn_scores_2 = query_2 @ keys.T
    print(attn_scores_2)
    # 三、计算注意力权重：注意力分数归一化
    # 通过嵌入维度的平方根(d_k ** 0.5)进行缩放解释了为什么这种自注意力机制也被称为缩放点积注意力机制。
    d_k = keys.shape[-1]
    attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim=-1)
    print(attn_weights_2)

    # 四、计算上下文向量：通过计算输入项的加权和获得。
    context_vec_2 = attn_weights_2 @ values
    print(context_vec_2)


def masked_attention():
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your (x^1)
         [0.55, 0.87, 0.66],  # journey (x^2)
         [0.57, 0.85, 0.64],  # starts (x^3)
         [0.22, 0.58, 0.33],  # with (x^4)
         [0.77, 0.25, 0.10],  # one (x^5)
         [0.05, 0.80, 0.55]]  # step (x^6)
    )
    # 输入嵌入维度
    d_in = inputs.shape[1]
    # 输出嵌入维度
    d_out = 2
    # 初始化三个权重矩阵
    torch.manual_seed(789)
    sa_v2 = SelfAttentionV2(d_in, d_out)
    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)
    values = sa_v2.W_value(inputs)
    attn_scores = queries @ keys.T
    context_length = attn_scores.shape[0]
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
    print(masked)
    attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)
    print(attn_weights)
    context_vec = attn_weights @ values
    print(context_vec)


def dropout_attention():
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your (x^1)
         [0.55, 0.87, 0.66],  # journey (x^2)
         [0.57, 0.85, 0.64],  # starts (x^3)
         [0.22, 0.58, 0.33],  # with (x^4)
         [0.77, 0.25, 0.10],  # one (x^5)
         [0.05, 0.80, 0.55]]  # step (x^6)
    )
    # 输入嵌入维度
    d_in = inputs.shape[1]
    # 输出嵌入维度
    d_out = 2
    # 初始化三个权重矩阵
    torch.manual_seed(789)
    sa_v2 = SelfAttentionV2(d_in, d_out)
    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)
    values = sa_v2.W_value(inputs)
    attn_scores = queries @ keys.T
    context_length = attn_scores.shape[0]
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
    print(masked)
    attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)
    print(attn_weights)
    context_vec = attn_weights @ values
    print(context_vec)


class SelfAttentionV1(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
        self.W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
        self.W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


class SelfAttentionV2(torch.nn.Module):
    def __init__(self, d_in, d_out, qv_bias=False):
        super().__init__()
        # 不同的权重初始化方式
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


class CausalAttention(torch.nn.Module):
    """
    因果注意力 (Causal Attention)：本质是「带掩码的自注意力」，通过上三角掩码
    屏蔽掉对未来位置 token的注意力计算，保证文本生成的逻辑合理性。
    因果注意力 vs 普通自注意力的唯一区别
    差一个掩码 mask！普通自注意力（Transformer 编码器）没有 mask，能看到整个序列的所有 token；
    因果注意力（Transformer 解码器）加了上三角掩码，只能看到当前及历史 token。
    CausalAttention类的完整执行流程：
    输入 token 特征
    → 线性投影生成 Q/K/V
    → 计算 token 间的注意力分数
    → 用上三角掩码屏蔽未来位置 → 分数缩放 + Softmax 得到注意力权重
    → Dropout 防止过拟合 → 权重加权求和 V 向量 → 输出融合语义的上下文向量
    """

    def __init__(self, d_in, d_out, context_length, dropout, qv_bias=False):
        super().__init__()
        # 定义Q/K/V三个线性投影层，将输入维度d_in映射到输出维度d_out
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qv_bias)
        # dropout层防止过拟合
        self.dropout = torch.nn.Dropout(dropout)
        # 核心：注册一个上三角掩码「因果掩码矩阵」，永久存在模型中，不参与梯度更新
        #  register_buffer会把mask放到和模型一致的设备（CPU/GPU），不用手动mask.to(device)
        self.register_buffer("mask",
                             torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # 获取输入张量的维度：批次、序列长度、输入特征维度
        keys = self.W_key(x)  # 计算K矩阵: [b, num_tokens, d_out]
        queries = self.W_query(x)  # 计算Q矩阵: [b, num_tokens, d_out]
        values = self.W_value(x)  # 计算V矩阵: [b, num_tokens, d_out]

        # K的转置把 [b, num_tokens, d_out] → [b, d_out, num_tokens]
        # Q和K的转置做矩阵乘法，得到注意力分数矩阵
        attn_scores = queries @ keys.transpose(1, 2)

        # 核心：用掩码把未来位置的注意力分数置为负无穷
        # 初始化的mask是[context_length, context_length]（最大序列长度），
        # 但实际输入的序列长度num_tokens可能小于最大值
        # [:num_tokens, :num_tokens]切片的作用：只取mask矩阵的前num_tokens行和列，适配当前输入的实际序列长度，避免维度不匹配
        # bool()将 mask 的 0/1 数值矩阵，转为布尔矩阵 True/False，masked_fill_ 只对True的位置生效
        # 负无穷 (-torch.inf) 注意力分数做softmax归一化，而 softmax(-∞) = 0
        # 未来位置的注意力权重会被置为 0，模型完全不会关注未来的 token，完美实现「因果性」
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        # 注意力分数缩放 + softmax归一化，得到注意力权重
        # keys.shape[-1] 就是d_out（Q/K 的维度）
        # 分数缩放：/ keys.shape[-1] ** 0.5 防止注意力分数过大导致的梯度消失
        # 因为 Q/K 的维度越大，内积计算出的分数值越大，softmax 后会趋近于 0 或 1，梯度会变的极小，模型无法训练。
        # 除以维度的平方根，能把分数值归一化到合理区间
        # 权重代表「当前 token 对每个历史 token 的关注比例」，
        # 比如第 3 个 token 的权重是[0.1, 0.7, 0.2]，说明模型主要关注第 2 个 token，次要关注第 3 个，少量关注第 1 个。
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        # 注意力权重 [b, num_tokens, num_tokens] 与 V 矩阵 [b, num_tokens, d_out] 做矩阵乘法
        # 输出形状：[b, num_tokens, d_out]
        # 核心意义：将注意力权重作为「权重系数」，对所有 token 的 V 向量做加权求和，得到每个 token 的「上下文特征向量」
        # 这个向量是融合了「当前 token + 所有历史 token 语义信息」的最终特征，
        # 也是整个注意力层的输出，会传入后续的 FeedForward 层继续计算。
        attn_weights = self.dropout(attn_weights)  # 对权重随机失活，防止过拟合

        # 注意力权重和V矩阵相乘，得到最终的上下文向量
        context_vec = attn_weights @ values
        return context_vec


def test_SelfAttentionV2():
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your (x^1)
         [0.55, 0.87, 0.66],  # journey (x^2)
         [0.57, 0.85, 0.64],  # starts (x^3)
         [0.22, 0.58, 0.33],  # with (x^4)
         [0.77, 0.25, 0.10],  # one (x^5)
         [0.05, 0.80, 0.55]]  # step (x^6)
    )
    # 输入嵌入维度
    dim_in = inputs.shape[1]
    # 输出嵌入维度
    dim_out = 2
    torch.manual_seed(123)
    sa_v1 = SelfAttentionV1(dim_in, dim_out)
    print(sa_v1(inputs))
    torch.manual_seed(789)
    sa_v2 = SelfAttentionV2(dim_in, dim_out)
    print(sa_v2(inputs))

def test_SelfAttention():
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your (x^1)
         [0.55, 0.87, 0.66],  # journey (x^2)
         [0.57, 0.85, 0.64],  # starts (x^3)
         [0.22, 0.58, 0.33],  # with (x^4)
         [0.77, 0.25, 0.10],  # one (x^5)
         [0.05, 0.80, 0.55]]  # step (x^6)
    )
    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape)
    # 输入嵌入维度
    dim_in = inputs.shape[1]
    # 输出嵌入维度
    dim_out = 2
    torch.manual_seed(123)
    context_length = batch.shape[1]
    ca = CausalAttention(dim_in, dim_out, context_length, 0.0)
    context_vecs = ca(batch)
    print("context_vecs.shape:", context_vecs.shape)


if __name__ == "__main__":
    test_SelfAttention()
