import re
import tiktoken
from gpt_dataset import GPTDatasetV1
from torch.utils.data import DataLoader

def create_vocab(text):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_words = sorted(set(preprocessed))
    all_words.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: integer for integer, token in enumerate(all_words)}
    return vocab

def create_vocab_with_verdict():
    return create_vocab(text=open("datas/the-verdict.txt", "r", encoding="utf-8").read())


def create_dataloader_with_verdict_v1(batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    return create_dataloader_v1(
        open("datas/the-verdict.txt", "r", encoding="utf-8").read(),
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

def create_dataloader_v1(txt, batch_size=4, max_length=256,
     stride=128, shuffle=True, drop_last=True,
     num_workers=0):
     tokenizer = tiktoken.get_encoding("gpt2")
     dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
     dataloader = DataLoader(
     dataset,
     batch_size=batch_size,
     shuffle=shuffle,
     drop_last=drop_last,
     num_workers=num_workers
     )
     return dataloader

def main():
    with open("datas/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)


if __name__ == '__main__':
    main()