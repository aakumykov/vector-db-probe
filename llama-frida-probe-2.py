import requests

# URL эндпоинта
url = "http://localhost:9931/v1/embeddings"

# Данные для отправки (словарь Python автоматически преобразуется в JSON)
payload = {
    "model": "FRIDA",
    "input": "ваш текст для получения эмбеддинга"
}

try:
    # Отправляем POST-запрос. Параметр json= автоматически добавляет заголовок Content-Type: application/json
    response = requests.post(url, json=payload)
    
    # Проверяем, что запрос прошел успешно (код 200)
    response.raise_for_status()
    
    # Получаем ответ в формате JSON
    result = response.json()
    
    # Извлекаем вектор эмбеддинга (в OpenAI-совместимом API он лежит в data[0].embedding)
    embedding = result["data"][0]["embedding"]
    
    print(f"Успешно получено! Размерность вектора: {len(embedding)}")
    print(f"Первые 5 значений вектора: {embedding[:5]}")

except requests.exceptions.RequestException as e:
    print(f"Ошибка при выполнении запроса: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Текст ошибки от сервера: {e.response.text}")
