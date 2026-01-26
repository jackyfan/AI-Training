from gpt_download import download_file

import json
import os
import requests


def download_and_load_file(url, file_path):
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            print("Downloading:", file_path)
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        print("Loading:", file_path)
        data = json.load(file)

    return data


if __name__ == "__main__":
    file_path = "datas\datasets\instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )

    data = download_and_load_file(url, file_path)
    print("Number of entries:", len(data))
