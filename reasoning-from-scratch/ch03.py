from pathlib import Path
import torch
from libs.qwen3 import (
    download_qwen3_small,
    Qwen3Tokenizer,
    Qwen3Model,
    QWEN_CONFIG_06_B
)
from libs.ch02 import (
    get_device
)
from libs.ch02_ex import (
    generate_text_basic_stream_cache
)


def load_model_and_tokenizer(which_model, device, use_compile, local_dir="qwen3"):
    if which_model == "base":
        download_qwen3_small(
            kind="base", tokenizer_only=False, out_dir=local_dir
        )
        tokenizer_path = Path(local_dir) / "tokenizer-base.json"
        model_path = Path(local_dir) / "qwen3-0.6B-base.pth"
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

    elif which_model == "reasoning":
        download_qwen3_small(
            kind="reasoning", tokenizer_only=False, out_dir=local_dir
        )
        tokenizer_path = Path(local_dir) / "tokenizer-reasoning.json"
        model_path = Path(local_dir) / "qwen3-0.6B-reasoning.pth"
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=tokenizer_path,
            apply_chat_template=True,
            add_generation_prompt=True,
            add_thinking=True,
        )

    else:
        raise ValueError(f"Invalid choice: which_model={which_model}")
    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    if use_compile:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)
    return model, tokenizer


def test_load_model_and_tokenizer():
    WHICH_MODEL = "base"
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(
        which_model=WHICH_MODEL,
        device=device,
        use_compile=False
    )
    print(model)


def mathematical_reasoner(prompt):
    WHICH_MODEL = "base"
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(
        which_model=WHICH_MODEL,
        device=device,
        use_compile=False
    )
    return generate_text_stream_concat(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=2048,
        verbose=True
    )


def generate_text_stream_concat(
        model, tokenizer, prompt, device, max_new_tokens,
        verbose=False,
):
    input_ids = torch.tensor(
        tokenizer.encode(prompt), device=device
    ).unsqueeze(0)

    generated_ids = []
    for token in generate_text_basic_stream_cache(
            model=model,
            token_ids=input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
    ):
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id.item())
        if verbose:
            print(
                tokenizer.decode(next_token_id.tolist()),
                end="",
                flush=True
            )
    return tokenizer.decode(generated_ids)


if __name__ == "__main__":
    # test_load_model_and_tokenizer()
    prompt = (
        r"If $a+b=3$ and $ab=\tfrac{13}{6}$, "
        r"what is the value of $a^2+b^2$?，使用中文回答"
    )
    mathematical_reasoner(prompt)
