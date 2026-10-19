#!.venv/bin/python
import requests

HOST="127.0.0.1"
PORT=9931
BASE_URL = f"http://{HOST}:{PORT}/v1"
MODEL_NAME="frida_guuf_bf16"

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

embedding = body[0]["embedding"]

print("размерность:", len(embedding))
print(f"длина вектора: {len(embedding)}")
c = 0
for i in embedding:
    print(f'{c}) {i[:5]} -- длина {len(i)} ({type(i)})')
    c+=1
