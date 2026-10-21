#!.venv/bin/python

import argparse
import math
from argparse import ArgumentError
import requests
from sentence_transformers.util import cos_sim

HOST="127.0.0.1"
PORT=9931
BASE_URL = f"http://{HOST}:{PORT}/v1"
MODEL_NAME="Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M"


def text2query_vector(text: str) -> list[float]:
    task_description = "Given a web search query, retrieve relevant passages that answer the query"
    input_text = f"Instruct: {task_description}\nQuery: {text}"
    print(f"query: {input_text}")
    return text2vector(input_text)


def text2document_vector(text: str) -> list[float]:
    print(f"document: {text}")
    return text2vector(text)


def text2vector(text: str) -> list:

    resp = requests.post(
        "http://127.0.0.1:9931/embeddings",
        json={
            "model": MODEL_NAME,
            "input": text,
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


parser = argparse.ArgumentParser(description="Создание векторной БД из txt файлов и семантический поиск.")
args, rest = parser.parse_known_args()


FIRST_TEXT=rest[0]
SECOND_TEXT=rest[1]


first_vector = text2vector(FIRST_TEXT)
second_vector = text2vector(SECOND_TEXT)

norm_first_vector = l2_normalize(first_vector)
norm_second_vector = l2_normalize(second_vector)

similarity = cos_sim(first_vector, second_vector)
norm_similarity = cos_sim(norm_first_vector, norm_second_vector)

print(f"'{FIRST_TEXT}' -- '{SECOND_TEXT}'")
print(f"нормализованных: {similarity.item():.4f}")
print(f"не-нормализованных: {norm_similarity.item():.4f}")