#!.venv/bin/python
import requests
import sys
import math
from sentence_transformers.util import cos_sim


HOST="127.0.0.1"
PORT=9931
BASE_URL = f"http://{HOST}:{PORT}/v1"
MODEL_NAME="frida_guuf_bf16"

def text2vector(text: str) -> list:
    resp = requests.post(
        "http://127.0.0.1:9931/embeddings",
        json={
            "model": MODEL_NAME,
            "input": "Какой-то текст для эмбеддинга",
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
    "рыба плавает в воде"
]

# for i, t in enumerate(text_list):
#     print(f'{i}) {t}')


for first in range(len(text_list) - 1):
    for second in range(first + 1, len(text_list)):
        # print(f'{first}-{second}')

        first_text = text_list[first]
        second_text = text_list[second]

        first_vector = text2vector(first_text)
        second_vector = text2vector(second_text)

        norm_first_vector = l2_normalize(first_vector)
        norm_second_vector = l2_normalize(second_vector)

        similarity = cos_sim(first_vector, second_vector)
        norm_similarity = cos_sim(norm_first_vector, norm_second_vector)

        print(f"сходство '{first_text}' | '{second_text}': не-Н:{similarity.item()}, Н:{norm_similarity.item()}")

    print('----')