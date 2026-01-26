from fontTools.ttLib.tables.ttProgram import instructions

from gpt_download import download_file

import json
import os
import requests
from torch.utils.data import Dataset
import tiktoken
import torch


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
def custom_collate_draft_1(batch,pad_token_id=50256,device="cpu"):
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
