import numpy as np

from gpt import GPTModel
import torch
import tiktoken
from gpt import generate_text_simple
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import os


def text_to_token_ids(text, tokenizer):
    encode = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    # 使用.unsqueeze(0)添加batch维度
    encode_tensor = torch.tensor(encode).unsqueeze(0)
    return encode_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    decode = tokenizer.decode(flat.tolist())
    return decode


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )

    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    token_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            token_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(token_seen)
                print(f"Epoch: {epoch}, Step: {global_step:06d}, "
                      f"Train loss: {train_loss:.3f}, "
                      f"Val loss: {val_loss:.3f}, "
                      f"Tokens seen: {token_seen}")
        generate_and_print_sample(model, tokenizer, device, start_context)
        return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weights.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text)
    model.train()


def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    # ===================== 外层循环：逐token生成 =====================
    # 循环max_new_tokens次，最多生成max_new_tokens个新token
    for _ in range(max_new_tokens):
        # ===================== 步骤1：截断输入序列（关键） =====================
        # idx[:, -context_size:]：只保留最后context_size个token（batch维度不变）
        # 原因：模型的位置嵌入只有context_size个，超出会导致索引越界报错
        idx_cond = idx[:, -context_size:]
        # ===================== 步骤2：模型前向预测（无梯度） =====================
        with torch.no_grad():  # 生成阶段无需计算梯度，节省显存+加速
            logits = model(idx_cond)  # 模型输出：[batch_size, seq_len, vocab_size]
            print("logits:\n", logits.shape)
        # ===================== 步骤3：提取最后一个token的预测结果 =====================
        # logits[:, -1, :]：只取序列最后一个token的logits（下一个token的预测概率）
        # 形状变为：[batch_size, vocab_size]
        logits = logits[:, -1, :]
        # ===================== 步骤4：Top-K采样（可选） =====================
        if top_k is not None:
            # 1. 取概率最高的top_k个token的最小值（比如top_k=50，取第50名的概率值）
            top_logits, _ = torch.topk(logits, top_k)  # top_logits形状：[batch_size, top_k]
            print("top_logits:\n", top_logits.shape)
            min_val = top_logits[:, -1]  # 每个batch的最小阈值：[batch_size]
            print("min_val:\n", min_val)
            # 2. 将所有低于该阈值的token的logits置为负无穷（后续softmax后概率为0）
            logits = torch.where(
                logits < min_val,  # 广播匹配logits形状：[batch_size, vocab_size]
                torch.tensor(float('-inf')).to(logits.device),  # 置为负无穷
                logits
            )
            print("where logits:\n", logits)
        # ===================== 步骤5：选择生成策略（贪心/采样） =====================
        if temperature > 0.0:
            # 温度采样：缩放logits后做softmax，再随机采样
            logits /= temperature
            probs = torch.softmax(logits, dim=-1)  # 转为概率分布：[batch_size, vocab_size]
            # 多项式采样：从概率分布中选1个token
            idx_next = torch.multinomial(probs, num_samples=1)  # 形状：[batch_size, 1]
        else:
            # 贪心解码（默认）：直接选概率最大的token（确定性生成）
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # 形状：[batch_size, 1]
        # ===================== 步骤6：检测终止符（可选） =====================
        if idx_next == eos_id:
            break
        # ===================== 步骤7：拼接新token到序列 =====================
        # 沿最后一维拼接：原序列 + 新token → 序列长度+1
        idx = torch.cat([idx, idx_next], dim=-1)
    return idx


# 检查两个张量或数组（left 和 right）是否具有相同的维度或形状
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left:{left.shape}",
                         f"Right:{right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


# 权重迁移 / 赋值工具
# 按 GPT 模型的结构（嵌入层→Transformer 块→输出层），
# 将params中的参数映射到gpt的对应属性上，
# 部分参数需要转置（.T）或拆分（np.split）后再赋值。
def load_weights_into_gpt(gpt, params):
    # 位置嵌入权重赋值（params["wpe"]是预训练的位置嵌入权重）
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    # 词嵌入权重赋值（params["wte"]是预训练的词嵌入权重）
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        # -------------------------- 自注意力层（Attention）权重拆分与赋值 --------------------------
        # 1. 拆分注意力层的权重：c_attn是合并的Q/K/V权重，按最后一维拆分为3份（对应Q、K、V）
        q_w, k_w, v_w = np.split(params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1)
        # 赋值Q/K/V的权重（.T转置：因为预训练权重的维度和自定义模型的权重维度可能相反）
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)
        # 2. 拆分注意力层的偏置：c_attn的偏置同样拆分为Q/K/V的偏置
        q_b, k_b, v_b = np.split(params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1)
        # 赋值Q/K/V的偏置（无需转置，因为偏置是一维向量）
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)

        # 3. 注意力层的输出投影层（out_proj）权重/偏置赋值
        gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(gpt.trf_blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])
        # -------------------------- 前馈网络（FFN）权重赋值 --------------------------
        # 1. FFN第一层（c_fc）权重/偏置赋值（转置原因同上）
        gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])

        # 2. FFN第二层（c_proj）权重赋值（注意：这里偏置错用了ln_1的g，可能是笔误！）
        gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(gpt.trf_blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"])
        # -------------------------- 层归一化（LayerNorm）参数赋值 --------------------------
        # 1. 第一个层归一化（norm1，注意力层后的归一化）的缩放（scale）和偏移（shift）
        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale,params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift,params["blocks"][b]["ln_1"]["b"])
        # 2. 第二个层归一化（norm2，前馈网络后的归一化）的缩放和偏移
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale,params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift,params["blocks"][b]["ln_2"]["b"])
    # 最终层归一化的缩放/偏移参数
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    # 输出头权重（复用词嵌入权重，这是GPT的经典设计：权重共享）
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

#加载GPT模型参数
def load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)


    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params

def test_generate_text_simple_with_temperature():
    vocab = {
        "closer": 0,
        "every": 1,
        "effort": 2,
        "forward": 3,
        "inches": 4,
        "moves": 5,
        "pizza": 6,
        "toward": 7,
        "you": 8,
    }
    inverse_vocab = {v: k for k, v in vocab.items()}
    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )
    probas = torch.softmax(next_token_logits, dim=0)
    next_token_id = torch.argmax(probas).item()
    print(inverse_vocab[next_token_id])
    torch.manual_seed(123)
    next_token_id = torch.multinomial(probas, num_samples=1).item()
    print(inverse_vocab[next_token_id])

    def print_sampled_tokens(probas):
        torch.manual_seed(123)
        sample = [torch.multinomial(probas, num_samples=1).item()
                  for i in range(1_000)]
        sampled_ids = torch.bincount(torch.tensor(sample))
        for i, freq in enumerate(sampled_ids):
            print(f"{freq} x {inverse_vocab[i]}")

    print_sampled_tokens(probas)

    def softmax_with_temperature(logits, temperature):
        scaled_logits = logits / temperature
        return torch.softmax(scaled_logits, dim=0)

    temperatures = [1, 0.1, 5]
    scaled_probas = [softmax_with_temperature(next_token_logits, T)
                     for T in temperatures]
    x = torch.arange(len(vocab))
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, T in enumerate(temperatures):
        rects = ax.bar(x + i * bar_width, scaled_probas[i],
                       bar_width, label=f'Temperature = {T}')
    ax.set_ylabel('Probability')
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.show()


def test_generate_text_simple_with_topk():
    vocab = {
        "closer": 0,
        "every": 1,
        "effort": 2,
        "forward": 3,
        "inches": 4,
        "moves": 5,
        "pizza": 6,
        "toward": 7,
        "you": 8,
    }
    inverse_vocab = {v: k for k, v in vocab.items()}
    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )
    probas = torch.softmax(next_token_logits, dim=0)
    next_token_id = torch.argmax(probas).item()
    print(inverse_vocab[next_token_id])
    torch.manual_seed(123)
    next_token_id = torch.multinomial(probas, num_samples=1).item()
    print(inverse_vocab[next_token_id])

    def print_sampled_tokens(probas):
        torch.manual_seed(123)
        sample = [torch.multinomial(probas, num_samples=1).item()
                  for i in range(1_000)]
        sampled_ids = torch.bincount(torch.tensor(sample))
        for i, freq in enumerate(sampled_ids):
            print(f"{freq} x {inverse_vocab[i]}")

    print_sampled_tokens(probas)

    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)
    print("Top logits:", top_logits)
    print("next_token_logits:", next_token_logits)
    min_val = top_logits[-1]
    print("Min val:", min_val)
    new_logits = torch.where(
        condition=next_token_logits < min_val,
        input=torch.tensor(float('-inf')),
        other=next_token_logits
    )
    print("New logits:", new_logits)
    print("New logits:", new_logits.shape)
    topk_probas = torch.softmax(new_logits, dim=0)
    print(topk_probas)

def main(gpt_config, input_prompt, model_size, device):

    settings, params = load_gpt2(model_size=model_size, models_dir="datas/models/gpt2")

    gpt = GPTModel(gpt_config)
    load_weights_into_gpt(gpt, params)
    gpt.to(device)
    gpt.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)

    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids(input_prompt, tokenizer).to(device),
        max_new_tokens=25,
        context_size=gpt_config["context_length"],
        top_k=50,
        temperature=1.5
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    torch.manual_seed(123)

    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves you"
    DEVICE = torch.device("cpu")

    print("PyTorch:", torch.__version__)
    print("Device:", DEVICE)

    BASE_CONFIG = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": True  # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    main(BASE_CONFIG, INPUT_PROMPT, model_size, DEVICE)
