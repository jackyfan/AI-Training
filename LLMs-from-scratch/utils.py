import re

def create_vocab(text):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_words = sorted(set(preprocessed))
    vocab = {token: integer for integer, token in enumerate(all_words)}
    return vocab

def create_vocab_with_verdict():
    return create_vocab(text=open("datas/the-verdict.txt", "r", encoding="utf-8").read())



def main():
    with open("datas/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    vocab = create_vocab(raw_text)
    print(len(vocab))
    for i, item in enumerate(vocab.items()):
        print(item)
        if i >= 50:
            break


if __name__ == '__main__':
    main()