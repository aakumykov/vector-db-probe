#!.venv/bin/python

from llama_cpp import Llama

# Загрузка GGUF модели
llm = Llama(
    model_path="path/to/Qwen3-Embedding-8B-GGUF/qwen3-embedding-8b-q4_k_m.gguf",
    embedding=True,
    n_ctx=8192 # или 32768, в зависимости от доступной памяти и версии модели
)

# 1. Формируем запрос с инструкцией
task_description = "Given a web search query, retrieve relevant passages that answer the query"
user_query = "Как настроить инструкцию в Qwen?"
query_text = f"Instruct: {task_description}\nQuery: {user_query}"

# 2. Документ передаем как есть
document_text = "Чтобы настроить инструкцию, добавьте префикс Instruct и Query к вашему тексту."

# 3. Получаем эмбеддинги
query_embedding = llm.create_embedding(query_text)["data"][0]["embedding"]
doc_embedding = llm.create_embedding(document_text)["data"][0]["embedding"]

# Далее вы можете рассчитать косинусное сходство между query_embedding и doc_embedding