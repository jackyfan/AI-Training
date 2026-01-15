from utils import create_dataloader_with_verdict_v1
import torch


def create_token_embedding():
    vocab_size = 50257
    output_dim = 256
    max_length = 4
    dataloader = create_dataloader_with_verdict_v1(
        batch_size=8, max_length=max_length,
        stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)

    # 词元嵌入层
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    # 词元 ID 嵌入 256 维的向量中
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)


    context_length = max_length
    # 位置嵌入层
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    # 位置嵌入张量
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(pos_embeddings.shape)

    # 输入嵌入张量
    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)





if __name__ == "__main__":
    create_token_embedding()