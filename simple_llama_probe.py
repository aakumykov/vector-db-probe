#!.venv/bin/python

import argparse
from argparse import ArgumentError
import requests
import sys
import math
from sentence_transformers.util import cos_sim

parser = argparse.ArgumentParser(description="Создание векторной БД из txt файлов и семантический поиск.")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-s", "--search", action='store_true', help="Режим поиска первого аргумента во втором")
group.add_argument("-p", "--paraphrase", action='store_true', help="Проверка на выражение другими словами")
# parser.add_argument("-1", type=str, dest="first_text", help="Текст №1")
# parser.add_argument("-2", type=str, dest="second_text", help="Текст №2")
args, rest = parser.parse_known_args()

MODE_NAME=""
MODE_OPERAND=""
FIRST_PREFIX=""
SECOND_PREFIX=""

FIRST_TEXT=rest[0]
SECOND_TEXT=rest[1]


if args.search:
    MODE_NAME = "поиск"
    MODE_OPERAND = "в"
    FIRST_PREFIX = "search_document"
    SECOND_PREFIX = "search_query"
elif args.paraphrase:
    MODE_NAME = "перефразировка"
    MODE_OPERAND = "и"
    FIRST_PREFIX = "paraphrase"
    SECOND_PREFIX = "paraphrase"
else:
    raise ArgumentError(args.search, f"Неизвестный режим работы ({parser.usage})")


HOST="127.0.0.1"
PORT=9931
BASE_URL = f"http://{HOST}:{PORT}/v1"
MODEL_NAME="frida_guuf_bf16"


def text2vector(text: str, prefix: str) -> list:

    # input_string = f"{prefix}: {text}"
    input_string = text
    print(f"input_string: {input_string}")

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


first_vector = text2vector(FIRST_TEXT, FIRST_PREFIX)
second_vector = text2vector(SECOND_TEXT, SECOND_PREFIX)

norm_first_vector = l2_normalize(first_vector)
norm_second_vector = l2_normalize(second_vector)

similarity = cos_sim(first_vector, second_vector)
norm_similarity = cos_sim(norm_first_vector, norm_second_vector)

print(f"{MODE_NAME}: '{FIRST_TEXT}' {MODE_OPERAND} '{SECOND_TEXT}'")
print(f"нормализованных: {similarity.item():.4f}")
print(f"не-нормализованных: {norm_similarity.item():.4f}")