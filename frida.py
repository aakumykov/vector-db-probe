#!.venv/bin/python
from argparse import ArgumentError
from unittest import case

import chromadb
import sys
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

def log_args():
    c=0
    for a in sys.argv:
        print(f'аргумент {c}: {sys.argv[c]}')
        c+=1


#log_args()


if 2 != len(sys.argv):
    print("Предоставьте имя коллекции для запросов: frida_test_1 или frida_test_2")
    sys.exit(1)


coll_name = sys.argv[1]


if "frida_test_2" == coll_name:
    encode_prefix = "search_document:"
    query_prefix = "search_query:"
else:
    encode_prefix = ""
    query_prefix = ""


log(f"coll_name:{coll_name}")
log(f"encode_prefix: {encode_prefix}")
log(f"query_prefix: {query_prefix}")


log_start("Авторизация в HuggingFace.....")
login_to_huggingface()
log_finish()


log_start("Загрузка модели ФРИДА.....")
frida_model = SentenceTransformer("ai-forever/FRIDA")
log_finish()


log_start("Создаю клиента БД.....")
chromadb_client = chromadb.PersistentClient(path="frida_db")
chromadb_collection = chromadb_client.get_or_create_collection(name=coll_name)
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
        one_embedding = frida_model.encode(f'{encode_prefix}{text}', convert_to_numpy=True)
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




# top_k = 5
# doc_preview_len = 500
#
# while True:
#     query_text = input("\nВаш запрос: ").strip()
#     if query_text.lower() in ['exit', 'quit', 'выход', 'q']:
#         break
#     if not query_text:
#         continue
#
#     query_embedding = frida_model.encode(f'{query_prefix}{query_text}').tolist()
#
#     results = chromadb_collection.query(
#         query_embeddings=query_embedding,
#         n_results=top_k
#     )
#
#     print(f"\n--- Топ-5 результатов ---")
#     for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
#         distance = results['distances'][0][i]
#         # Красиво обрезаем длинный текст для вывода
#         preview = doc[:doc_preview_len].replace('\n', ' ') + "..." if len(doc) > doc_preview_len else doc.replace('\n', ' ')
#         print(f"[{i + 1}] Файл: {meta['file_name']} (Дистанция: {distance:.4f})")
#         print(f"    Текст: {preview}\n")
