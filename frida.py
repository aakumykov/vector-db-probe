#!.venv/bin/python

import chromadb
from chromadb.types import Collection
from sentence_transformers import SentenceTransformer
from sympy import true, false

from for_text_in_files import for_text_in_files
from login_to_huggingface import login_to_huggingface
from collections.abc import Callable
from pathlib import Path

def log(s: str):
    print(s)

def log_start(s: str):
    print(s, end='')

def log_finish(s: str = "готово"):
    print(s)


log_start("Авторизация в HuggingFace.....")
login_to_huggingface()
log_finish()


log_start("Загрузка модели ФРИДА.....")
frida_model = SentenceTransformer("ai-forever/FRIDA")
log_finish()


log_start("Создаю клиента БД.....")
chromadb_client = chromadb.Client(database="frida_db")
chromadb_collection = chromadb_client.create_collection(name="frida_test_1")
log_finish()


log("Извлекаю данные из файлов.....")
embeddings = []
ids = []
raw_documents = []
metadatas = []

for p in Path("txt").iterdir():
    if p.is_file() and p.suffix == '.txt':
        text = p.read_text(encoding='utf-8')
        file_name = p.name
        id = file_name
        metadata = { "file_name": file_name, "text_length": len(text) }

        log_start(f"Кодирую {file_name}.....")
        one_embedding = frida_model.encode(text, convert_to_numpy=True)
        log_finish()

        # TODO: convert_to_tensor=True
        embeddings.append(one_embedding)
        ids.append(id)
        metadatas.append(metadata)
        raw_documents.append(text)


log_start("Сохраняю все коды в БД.....")
chromadb_collection.upsert(
    embeddings=embeddings,
    ids=ids,
    documents=raw_documents,
    metadatas=metadatas,
)
log_finish(f"Успешно добавлено {len(embeddings)} документов в векторную БД.")

