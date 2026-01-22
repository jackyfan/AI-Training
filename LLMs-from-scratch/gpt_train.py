from gpt import GPTModel
import torch
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from gpt import generate_text_simple
from utils import create_dataloader_v1, create_vocab_with_verdict


def text_to_token_ids(text, tokenizer):
    encode = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    # 使用.unsqueeze(0)添加batch维度
    encode_tensor = torch.tensor(encode).unsqueeze(0)
    return encode_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    decode = tokenizer.decode(flat.tolist())
    return decode

# 计算单个批次的损失
def calc_loss_batch(input_batch, target_batch, model, device):
    # to(devices)可以将数据转移到GPU上
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

# 计算数据加载器采样的所有批次的损失
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        # 没有指定批次数量、遍历数据加载器的所有批次
        num_batches = len(data_loader)
    else:
        # 取指定批次与数据加载器批次的较小值
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            # 每个批次的损失求和
            total_loss += loss.item()
        else:
            break
    #对所有批次的损失求平均值
    return total_loss / num_batches


def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # ===================== 初始化监控变量 =====================
    # 训练损失、验证损失、已处理的所见的词元tokens
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    # ===================== 外层：遍历训练轮数 =====================
    for epoch in range(num_epochs):
        # 核心：将模型设为「训练模式」（启用Dropout、BatchNorm等训练专属层）
        model.train()
        # ===================== 内层：遍历训练批次 =====================
        for input_batch, target_batch in train_loader:
            # 1. 重置优化器梯度（PyTorch梯度累加，必须手动清零，否则会叠加之前批次的梯度）
            optimizer.zero_grad()
            # 2.计算当前批次的损失（核心：调用calc_loss_batch函数，完成前向传播）
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # 3. 反向传播：计算模型参数的梯度（从loss反向推导每个参数的梯度）
            loss.backward()
            # 4. 优化器更新：用梯度更新模型参数（如AdamW根据梯度调整权重）
            optimizer.step() # numel()返回张量总元素数（即批次词元数）
            # 5. 累计统计：更新已处理词元数、全局步数
            tokens_seen += input_batch.numel()
            global_step += 1
            # ===================== 定期验证：监控模型收敛 =====================
            # 每eval_freq个step，执行一次训练集/验证集损失评估
            if global_step % eval_freq == 0:
                # 调用evaluate_model：计算指定迭代数（eval_iter）的平均训练/验证损失
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)

                # 记录损失和词元数（用于后续分析，比如绘制损失曲线）
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                # 打印监控信息：轮数、步数、训练/验证损失（保留3位小数）
                print(f"Epoch: {epoch+1}, Step: {global_step:06d}, "
                      f"Train loss: {train_loss:.3f}, "
                      f"Val loss: {val_loss:.3f}")
        # 训练轮结束后，用指定的start_context生成文本，直观查看模型训练效果
        generate_and_print_sample(model, tokenizer, device, start_context)
    # ===================== 训练结束：返回监控数据 =====================
    # 返回训练/验证损失列表、词元数列表（用于后续分析，比如绘制损失曲线）
    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # ===================== 第一步：切换模型为「评估模式」 =====================
    # 核心：禁用训练专属的层（如Dropout、BatchNorm），确保评估结果稳定
    model.eval()
    # ===================== 第二步：关闭梯度计算（关键优化） =====================
    with torch.no_grad():
        # 1. 计算训练集平均损失：调用calc_loss_loader，仅用前eval_iter个批次
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        # 2. 计算验证集平均损失
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    # ===================== 第三步：切回「训练模式」 =====================
    # 核心：不影响后续训练（比如下一个批次的Dropout需要重新启用）
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    # ===================== 第一步：切换模型为「评估模式」 =====================
    # 核心：禁用训练专属的层（如Dropout、BatchNorm），确保评估结果稳定
    model.eval()
    # ===================== 第二步：获取模型的上下文长度 =====================
    # model.pos_emb：模型的位置嵌入层（Transformer核心组件）
    # pos_emb.weight.shape[0]：位置嵌入的行数 = 模型支持的最大上下文长度（context_size）
    # 比如context_size=512，说明模型最多处理512个token的上下文
    context_size = model.pos_emb.weight.shape[0]
    # ===================== 第三步：编码起始文本为token ID =====================
    # text_to_token_ids：自定义函数，将字符串（start_context）转为token ID张量
    # .to(device)：将张量移至GPU/CPU（与模型设备一致，避免数据位置不匹配报错）
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    # ===================== 第四步：生成新文本（核心） =====================
    with torch.no_grad(): # 关闭梯度计算：生成阶段无需更新参数，节省显存+加速
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,# 起始token ID（提示词的编码结果）
            max_new_tokens=50,# 最多生成50个新token
            context_size=context_size# 模型支持的最大上下文长度（防止越界）
        )
    # ===================== 第五步：解码token ID为可读文本 =====================
    # token_ids_to_text：自定义函数，将生成的token ID张量转回字符串
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    # ===================== 第六步：格式化并打印生成结果 =====================
    # .replace("\n"," ")：将换行符替换为空格，避免打印时换行混乱，方便查看完整文本
    print(decoded_text.replace("\n"," "))
    # ===================== 第三步：切回「训练模式」 =====================
    # 核心：不影响后续训练（比如下一个批次的Dropout需要重新启用）
    model.train()

def test_generate_text_simple(model,gpt_cfg):
    tokenizer = tiktoken.get_encoding("gpt2")
    ids = text_to_token_ids("Every effort moves you", tokenizer)
    token_ids = generate_text_simple(model=model,idx=ids,
                                     max_new_tokens=10,
                                     context_size=gpt_cfg["context_length"])
    print("Out text:\n",token_ids_to_text(token_ids, tokenizer))

def calc_text_token_length(text):
    tokenizer = tiktoken.get_encoding("gpt2")
    total_characters = len(text)
    total_tokens = len(tokenizer.encode(text_data))
    print("Characters:", total_characters)
    print("Tokens:", total_tokens)

def plot_losses(epochs_seen,tokens_seen,train_losses,val_losses):
    fig,ax1 = plt.s


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
    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)
    gpt_model = GPTModel(GPT_CONFIG_124M)
    # test_generate_text_simple(model, GPT_CONFIG_124M)
    file_path = "datas/the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    # calc_text_token_length(text_data)
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)
    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt_model.to(device)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, gpt_model, device)
        val_loss = calc_loss_loader(val_loader, gpt_model, device)
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)

    optimizer = torch.optim.AdamW(
        gpt_model.parameters(),
        lr=0.0004, weight_decay=0.1
    )
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        gpt_model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )
