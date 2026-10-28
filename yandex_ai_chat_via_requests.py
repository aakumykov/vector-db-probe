import requests
import json
from read_file_from_home import read_file_from_home

# 1. Настройки запроса
API_KEY = read_file_from_home('.yandex_ai_token')
URL = "https://ai.api.cloud.yandex.net/v1/responses"
YANDEX_FOLDER_ID = "b1g2nef5l3idehorhcup"
YANDEX_MODEL = "yandexgpt-5-lite"

# 2. Заголовки (Headers)
# Для OpenAI-совместимого API авторизация всегда идет через Bearer токен
headers = {
    "Authorization": f"Api-Key {API_KEY}",
    "Content-Type": "application/json"
}

# 3. Тело запроса (Payload)
# Формируем JSON в формате OpenAI Chat Completion
payload = {
    "model": f"gpt://{YANDEX_FOLDER_ID}/{YANDEX_MODEL}",
    "temperature": 0.0,
    "max_output_tokens": 512,
    "input": "Придумай 3 необычные идеи для стартапа в сфере путешествий."
}

# 4. Отправка запроса
try:
    response = requests.post(URL, headers=headers, json=payload, timeout=60)
    
    # 5. Обработка ответа
    if response.status_code == 200:
        result = response.json()
        
        # Извлекаем текст ответа из структуры JSON
        # assistant_message = result["choices"][0]["message"]["content"]
        assistant_message = result['output'][0]['content'][0]['text']
        print("Ответ модели:\n", assistant_message)
        
        # Также можно посмотреть на использование токенов
        usage = result.get("usage", {})
        print(f"\nИспользовано токенов: {usage.get('total_tokens', 'N/A')}")
        
    else:
        print(f"Ошибка API. Статус код: {response.status_code}")
        print("Детали ошибки:", response.text)

except requests.exceptions.RequestException as e:
    print(f"Произошла сетевая ошибка: {e}")