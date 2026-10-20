#!.venv/bin/python

import json
import requests
import numpy as np


OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "hf.co/evilfreelancer/FRIDA-GGUF:BF16"


def get_embedding(text):
    payload = {
        "model": MODEL_NAME,
        "input": text
    }

    response = requests.post(
        f"{OLLAMA_HOST}/api/embed",
        data=json.dumps(payload, ensure_ascii=False),
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    response.raise_for_status()
    return np.array(response.json()["embeddings"][0])


def normalize(vectors):
    vectors = np.atleast_2d(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def cosine_diag_similarity(a, b):
    return np.sum(a * b, axis=1)


inputs = [
    #
    "paraphrase: В Ярославской области разрешили работу бань, но без посетителей",
    "categorize_entailment: Женщину доставили в больницу за ее жизнь сейчас борются врачи.",
    "search_query: Сколько программистов нужно, чтобы вкрутить лампочку?",
    #
    "paraphrase: Ярославским баням разрешили работать без посетителей",
    "categorize_entailment: Женщину спасают врачи.",
    "search_document: Чтобы вкрутить лампочку нужно три программиста.",
]
size = int(len(inputs)/2)

embeddings = normalize(np.array([get_embedding(text) for text in inputs]))
sim_scores = cosine_diag_similarity(embeddings[:size], embeddings[size:])
print(sim_scores.tolist())

