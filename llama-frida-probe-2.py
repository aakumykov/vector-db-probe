#!.venv/bin/python
import sys

import requests
from pygments.styles import paraiso_dark
from sentence_transformers.util import cos_sim
import math
import argparse


# MODEL_NAME = "evilfreelancer/FRIDA-GGUF:BF16"
# MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"
MODEL_NAME = "Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M"
URL = "http://localhost:9931/v1/embeddings"


def l2_normalize(vector: list[float]) -> list[float]:
    """L2-нормализация вектора"""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


def text2vector(text: str) -> list[float]:

    print(f"text2vector: {text}")

    payload = {
        "model": MODEL_NAME,
        "input": text
    }

    # Отправляем POST-запрос. Параметр json= автоматически добавляет заголовок Content-Type: application/json
    response = requests.post(URL, json=payload)

    # Проверяем, что запрос прошел успешно (код 200)
    response.raise_for_status()

    # Получаем ответ в формате JSON
    result = response.json()

    # Извлекаем вектор эмбеддинга (в OpenAI-совместимом API он лежит в data[0].embedding)
    embedding = result["data"][0]["embedding"]

    return embedding


parser = argparse.ArgumentParser(description="Векторное сходство двух текстов")
parser.add_argument("-1", "--first", type=str, required=False, help="Первый текст")
parser.add_argument("-2", "--second", type=str, required=False, help="Вторй текст")
args, rest = parser.parse_known_args()


FIRST_TEXT=""
SECOND_TEXT=""

if args.first:
    FIRST_TEXT = args.first

if args.second:
    SECOND_TEXT = args.second

if not args.first and not args.second:
    FIRST_TEXT=rest[0]
    SECOND_TEXT=rest[1]

if not FIRST_TEXT and not SECOND_TEXT:
    parser.print_help()
    sys.exit(1)


first_vector = text2vector(FIRST_TEXT)
second_vector = text2vector(SECOND_TEXT)

norm_first_vector = l2_normalize(first_vector)
norm_second_vector = l2_normalize(second_vector)

similarity = cos_sim(first_vector, second_vector)
norm_similarity = cos_sim(norm_first_vector, norm_second_vector)

print(f"'{FIRST_TEXT}' -- '{SECOND_TEXT}'")
print(f"нормализованных: {similarity.item():.4f}")
print(f"не-нормализованных: {norm_similarity.item():.4f}")
