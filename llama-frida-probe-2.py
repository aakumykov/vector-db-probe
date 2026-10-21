#!.venv/bin/python

import requests
from sentence_transformers.util import cos_sim
import math
import argparse


MODEL_NAME = "evilfreelancer/FRIDA-GGUF:BF16"
URL = "http://localhost:9931/v1/embeddings"


def l2_normalize(vector: list[float]) -> list[float]:
    """L2-нормализация вектора"""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


def text2vector(text: str) -> list[float]:
    payload = {
        "model": MODEL_NAME,
        "input": "ваш текст для получения эмбеддинга"
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
