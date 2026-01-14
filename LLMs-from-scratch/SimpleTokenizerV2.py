import re
import utils


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text):
        """处理输入文本 ，根据词汇表转化为词元ID"""
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        """将词元ID转化为文本"""
        text = " ".join([self.int_to_str[id] for id in ids])
        text = re.sub(r'([,.?_!"()\']|--)', r'\1', text)
        return text


def main():
    vocab = utils.create_vocab_with_verdict()
    tokenizer = SimpleTokenizerV2(vocab)
    text = """"It's the last he painted, you know," 
     Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))
    unknown_text = "Hello, do you like tea?"
    # 未知的单词，会报错
    print( tokenizer.encode(unknown_text))



if __name__ == "__main__":
    main()
