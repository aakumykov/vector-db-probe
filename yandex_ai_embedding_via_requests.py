import requests
import json
from read_file_from_home import read_file_from_home

YANDEX_API_KEY = read_file_from_home('.yandex_ai_token')
EMBED_URL = "https://ai.api.cloud.yandex.net:443/foundationModels/v1/textEmbedding"
YANDEX_FOLDER_ID = "b1g2nef5l3idehorhcup"
YANDEX_MODEL = "text-embeddings-v2-doc"

# headers = {
#     "Authorization": f"Api-Key {YANDEX_API_KEY}",
#     "Content-Type": "application/json"
# }
headers = {
    "Content-Type": "application/json", 
    "Authorization": f"Bearer {YANDEX_API_KEY}", 
    "x-folder-id": f"{YANDEX_FOLDER_ID}"
}

doc_uri   = f"emb://{YANDEX_FOLDER_ID}/text-embeddings-v2-doc/latest"
query_uri = f"emb://{YANDEX_FOLDER_ID}/text-search-query/latest"

model_uri = doc_uri
text = "Хуй проссышь!"

payload = {
    "modelUri": model_uri,
    "text": text
}

# 4. Отправка запроса
try:
    response = requests.post(EMBED_URL, headers=headers, json=payload, timeout=60)
    
    # 5. Обработка ответа
    if response.status_code == 200:
        result = response.json()
        
        # Извлекаем текст ответа из структуры JSON
        # assistant_message = result["choices"][0]["message"]["content"]
        assistant_message = result['embedding']
        print("Ответ модели:\n", assistant_message)
        
        # Также можно посмотреть на использование токенов
        usage = result.get("numTokens", {})
        print(f"\nИспользовано токенов: {usage}")
        
    else:
        print(f"Ошибка API. Статус код: {response.status_code}")
        print("Детали ошибки:", response.text)

except requests.exceptions.RequestException as e:
    print(f"Произошла сетевая ошибка: {e}")