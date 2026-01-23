from gpt import GPTModel
import torch
import tiktoken
from gpt import generate_text_simple
import matplotlib.pyplot as plt


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
        with torch.no_grad(): # 生成阶段无需计算梯度，节省显存+加速
            logits = model(idx_cond) # 模型输出：[batch_size, seq_len, vocab_size]
            print("logits:\n", logits.shape)
        # ===================== 步骤3：提取最后一个token的预测结果 =====================
        # logits[:, -1, :]：只取序列最后一个token的logits（下一个token的预测概率）
        # 形状变为：[batch_size, vocab_size]
        logits = logits[:, -1, :]
        # ===================== 步骤4：Top-K采样（可选） =====================
        if top_k is not None:
            # 1. 取概率最高的top_k个token的最小值（比如top_k=50，取第50名的概率值）
            top_logits, _ = torch.topk(logits, top_k)# top_logits形状：[batch_size, top_k]
            print("top_logits:\n", top_logits.shape)
            min_val = top_logits[:, -1] # 每个batch的最小阈值：[batch_size]
            print("min_val:\n",min_val)
            # 2. 将所有低于该阈值的token的logits置为负无穷（后续softmax后概率为0）
            logits = torch.where(
                logits < min_val,# 广播匹配logits形状：[batch_size, vocab_size]
                torch.tensor(float('-inf')).to(logits.device),  # 置为负无穷
                logits
            )
            print("where logits:\n",logits)
        # ===================== 步骤5：选择生成策略（贪心/采样） =====================
        if temperature > 0.0:
            # 温度采样：缩放logits后做softmax，再随机采样
            logits /= temperature
            probs = torch.softmax(logits, dim=-1)# 转为概率分布：[batch_size, vocab_size]
            # 多项式采样：从概率分布中选1个token
            idx_next = torch.multinomial(probs, num_samples=1)# 形状：[batch_size, 1]
        else:
            # 贪心解码（默认）：直接选概率最大的token（确定性生成）
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)# 形状：[batch_size, 1]
        # ===================== 步骤6：检测终止符（可选） =====================
        if idx_next == eos_id:
            break
        # ===================== 步骤7：拼接新token到序列 =====================
        # 沿最后一维拼接：原序列 + 新token → 序列长度+1
        idx = torch.cat([idx, idx_next], dim=-1)
    return idx


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


if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # 词汇表大小
        "context_length": 256,  # 上下文长度
        "emb_dim": 768,  # 嵌入维度
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to("cpu")
    model.eval()
    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=0.5
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    test_generate_text_simple_with_topk()