import re
import utils


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_id = vocab
        self.id_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text):
        """处理输入文本 ，根据词汇表转化为词元ID"""
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_id[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        """将词元ID转化为文本"""
        text = " ".join([self.id_to_str[id] for id in ids])
        text = re.sub(r'([,.?_!"()\']|--)', r'\1', text)
        return text

