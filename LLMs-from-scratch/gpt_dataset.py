import torch
from torch.utils.data import Dataset


class GPTDatasetV1(Dataset):
    def __init__(self, texts, tokenizer, max_length, stride):
        self.intput_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(texts)
        for i in range(0, len(token_ids) - max_length, stride):
            input_ids = token_ids[i:i + max_length]
            target_ids = token_ids[i + 1:i + 1 + max_length]
            self.intput_ids.append(torch.tensor(input_ids))
            self.target_ids.append(torch.tensor(target_ids))

    def __len__(self):
        return len(self.intput_ids)

    def __getitem__(self, index):
        return self.intput_ids[index], self.target_ids[index]
