#!.venv/bin/python

import argparse
from argparse import ArgumentError
import requests
import sys
import math
from sentence_transformers.util import cos_sim


# parser = argparse.ArgumentParser(description="Создание векторной БД из txt файлов и семантический поиск.")
# parser.add_argument("--prefix", action='store_true', type=bool, default=False, dest="use_prefix", help="Использовать префиксы (search_document, search_query)")
# args = parser.parse_args()
#
# use_prefix = args.use_prefix

DOCUMENT_PREFIX="search_document"
QUERY_PREFIX="search_query"
PARAPHRASE_PREFIX="paraphrase"

HOST="127.0.0.1"
PORT=9931
BASE_URL = f"http://{HOST}:{PORT}/v1"
MODEL_NAME="frida_guuf_bf16"


def text2vector(text: str, prefix: str) -> list:
    input_string = f"{prefix}:{text}"
    # print(input_string)

    resp = requests.post(
        "http://127.0.0.1:9931/embeddings",
        json={
            "model": MODEL_NAME,
            "input": input_string,
            "encoding_format": "float",
        },
        timeout=120,
    )
    resp.raise_for_status()
    body = resp.json()

    return body[0]["embedding"][0]


def l2_normalize(vector: list[float]) -> list[float]:
    """L2-нормализация вектора"""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


text_list = [
    "Кошка сидит на окне",
    "На окне сидит кошка",
    # "кошка на окне сидит",
    # "на окне кошка сидит",
    # "сидит на окне кошка",

    "рыба плавает в воде",

    "программист работает за компьютером",

    "завершена уборочная кампания",

    "qwerty",
    "qwerty",
    "йцукен",
    "abcdef",
    "ABCDEF"
]

# for i, t in enumerate(text_list):
#     print(f'{i}) {t}')


for first in range(len(text_list) - 1):
    for second in range(first + 1, len(text_list)):
        # print(f'{first}-{second}')

        first_text = text_list[first]
        second_text = text_list[second]
        # print(f'{first}: {first_text} -- {second}: {second_text}')

        first_vector = text2vector(first_text, PARAPHRASE_PREFIX)
        second_vector = text2vector(second_text, PARAPHRASE_PREFIX)

        norm_first_vector = l2_normalize(first_vector)
        norm_second_vector = l2_normalize(second_vector)

        similarity = cos_sim(first_vector, second_vector)
        norm_similarity = cos_sim(norm_first_vector, norm_second_vector)

        print(f"сходство '{first_text}' | '{second_text}': не-Н:{similarity.item():.4f}, Н:{norm_similarity.item():.4f}")

    print('----')