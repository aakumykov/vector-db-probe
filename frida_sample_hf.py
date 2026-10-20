#!.venv/bin/python

import json
import requests
import numpy as np

OLLAMA_HOST = "http://localhost:9931"
MODEL_NAME = "frida_guuf_bf16"


def get_embedding(text):
    payload = {
        "model": MODEL_NAME,
        "input": text
    }

    response = requests.post(
        f"{OLLAMA_HOST}/v1/embeddings",
        data=json.dumps(payload, ensure_ascii=False),
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    response.raise_for_status()
    e = response.json()["data"][0]["embedding"]
    return np.array(e)


def normalize(vectors):
    vectors = np.atleast_2d(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def cosine_diag_similarity(a, b):
    return np.sum(a * b, axis=1)


inputs = [
    #
    "search_query: false",
    #
    "search_document: true",
]
size = int(len(inputs)/2)

embeddings = normalize(np.array([get_embedding(text) for text in inputs]))
sim_scores = cosine_diag_similarity(embeddings[:size], embeddings[size:])
print(sim_scores.tolist())
