import json
import os
import requests
from torch.utils.data import Dataset,DataLoader
import tiktoken
import torch
from functools import partial
from gpt_download import download_and_load_gpt2
from gpt_generate import GPTModel,load_weights_into_gpt


def download_and_load_file(url, file_path):
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_text = []
        for entry in self.data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_text.append(tokenizer(full_text))

    def __len__(self):
        return len(self.encoded_text)

    def __getitem__(self, idx):
        return self.encoded_text[idx]


# 聚合函数来实现填充自动补齐
def custom_collate_draft_1(batch, pad_token_id=50256, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst = []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
                new_item + [pad_token_id] *
                (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor


# 函数的核心价值是适配语言模型的自回归训练逻辑：
# 解决变长序列的批次拼接问题（填充到统一长度）；
# 生成 “输入预测下一个词元、目标是该词元” 的训练对；
# 保证张量形状一致、设备匹配，可直接用于模型训练。
def custom_collate_draft_2(batch, pad_token_id=50256, device="cpu"):
    # 计算批次最大长度
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        # 复制原序列
        new_item += [pad_token_id]
        padded = (
                new_item + [pad_token_id] *
                (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

#聚合函数，以将目标列表中 ID 为 50256 的词元替换为-100
def custom_collate_fn(batch,pad_token_id=50256,
        ignore_index=-100,allowed_max_length=None,device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
                new_item + [pad_token_id] *
                (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


if __name__ == "__main__":
    file_path = "datas\datasets\instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )

    data = download_and_load_file(url, file_path)
    print("Number of entries:", len(data))

    model_input = format_input(data[999])
    desired_response = f"\n\n### Response:\n{data[50]['output']}"
    print(model_input + desired_response)

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    val_data = data[train_portion:train_portion + val_portion]
    test_data = data[train_portion + val_portion:]

    print("len of train entries:", len(train_data))
    print("len of val entries:", len(val_data))
    print("len of test entries:", len(test_data))

    tokenizer = tiktoken.get_encoding("gpt2")

    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    batch = (
        inputs_1,
        inputs_2,
        inputs_3
    )
    print(custom_collate_draft_1(batch))

    inputs, targets = custom_collate_draft_2(batch)
    print(inputs)
    print(targets)

    inputs, targets = custom_collate_fn(batch)
    print(inputs)
    print(targets)
    device = torch.device("cpu")
    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=1024
    )
    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)
    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )
    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
         shuffle=False,
         drop_last=False,
         num_workers=num_workers
        )
    print("Train loader:")
    for inputs, targets in train_loader:
        print(inputs.shape, targets.shape)

    BASE_CONFIG = {
        "vocab_size": 50257,  # 词汇表大小
        "context_length": 1024,  # 上下文长度
        "drop_rate": 0.0,  # dropout 率
        "qkv_bias": True  # 查询-键-值偏置
    }
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir="gpt2"
    )
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
