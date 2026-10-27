import json
import requests
import numpy as np

LLAMA_HOST = "http://localhost:9931"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"
# MODEL_NAME = "Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M"

URL = f"{LLAMA_HOST}/v1/embeddings"

def get_embedding(text):
    payload = {
        "model": MODEL_NAME,
        "input": text
    }

    response = requests.post(
        URL,
        json=payload,
        # data=json.dumps(payload, ensure_ascii=False),
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    response.raise_for_status()
    result = response.json()
    embedding = result["data"][0]["embedding"]
    return np.array(embedding)


def normalize(vectors):
    vectors = np.atleast_2d(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def cosine_diag_similarity(a, b):
    return np.sum(a * b, axis=1)


# inputs = [
#     #
#     "paraphrase: В Ярославской области разрешили работу бань, но без посетителей",
#     "categorize_entailment: Женщину доставили в больницу за ее жизнь сейчас борются врачи.",
#     "search_query: Сколько программистов нужно, чтобы вкрутить лампочку?",
#     #
#     "paraphrase: Ярославским баням разрешили работать без посетителей",
#     "categorize_entailment: Женщину спасают врачи.",
#     "search_document: Чтобы вкрутить лампочку нужно три программиста.",
# ]
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

# embeddings = normalize(np.array([get_embedding(text) for text in inputs]))
# sim_scores = cosine_diag_similarity(embeddings[:size], embeddings[size:])
# print(sim_scores.tolist())


text1 = "qwerty"
text2 = "5"

p_text1 = f"paraphrase: {text1}"
p_text2 = f"paraphrase: {text2}"

vec1 = normalize(get_embedding(text1))
vec2 = normalize(get_embedding(text2))

p_vec1 = normalize(get_embedding(p_text1))
p_vec2 = normalize(get_embedding(p_text2))

print(f"сходство между '{text1}' и '{text2}': {cosine_diag_similarity(vec1, vec2)}")
print(f"сходство между '{p_text1}' и '{p_text2}': {cosine_diag_similarity(p_vec1, p_vec2)}")