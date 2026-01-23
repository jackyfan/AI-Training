from gpt_download import download_and_load_gpt2
from gpt import GPTModel

settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="datas/models/gpt2"
)

print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
GPT_CONFIG_124M = {
        "vocab_size": 50257,  # 词汇表大小
        "context_length": 256,  # 上下文长度
        "emb_dim": 768,  # 嵌入维度
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})
gpt = GPTModel(NEW_CONFIG)
gpt.eval()