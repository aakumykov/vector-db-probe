#!.venv/bin/python

import re
from argparse import ArgumentError
from typing import Any

import requests
import argparse
import os
import sys
from enum import Enum
from os.path import expanduser
from pathlib import Path

import chromadb
# from chromadb import ClientAPI


SERVER_HOST="localhost"
SERVER_PORT=9931
QUERY_PATH="/v1/embeddings"
MODELS_PATH="/v1/models"
URL_BASE = f"http://{SERVER_HOST}:{SERVER_PORT}"
MODELS_URL = f"{URL_BASE}{MODELS_PATH}"
EMB_URL = f"{URL_BASE}{QUERY_PATH}"

DEFAULT_RESULTS_COUNT=5

DB_PATH="db"
COLLECTION_NAME: str
DB_CLIENT: Any #: ClientAPI
DB_COLLECTION: Any

MODEL_NAME: str

TXT_DIR_PATH: Path


parser = argparse.ArgumentParser(description="Создание векторной БД из txt файлов и семантический поиск.")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-l", "--list", action='store_true', help="Вывести список доступных моделей")
group.add_argument("--load", dest="dir", type=str, help="Путь к каталогу с txt файлами")
group.add_argument("-q", "--query", type=str, help="Текст запроса к БД")

parser.add_argument("-m", "--model", type=str, required=False, help="Название модели")
parser.add_argument("-c", "--count", type=int, required=False, default=DEFAULT_RESULTS_COUNT, help=f"Количество результатов поиска (по умолчанию: {DEFAULT_RESULTS_COUNT})")
args = parser.parse_args()


def log_d(text: str):
    print(text)


def list_models_on_server():
    response = requests.get(MODELS_URL)
    response.raise_for_status()
    model_ids = [model["id"] for model in response.json()["data"]]
    print(f"===== Models on {SERVER_HOST}: =====")
    for m in model_ids:
        print(m)


def check_model_argument():
    if not args.model:
            print("He yka3aHo umR moDejlu")
            exit(1)
    
    global MODEL_NAME
    MODEL_NAME = args.model
    log_d(f"MODEL_NAME: {MODEL_NAME}")

    global COLLECTION_NAME
    COLLECTION_NAME = re.sub("[^0-9a-zA-Z]+", "_", MODEL_NAME)
    log_d(f"COLLECTION_NAME: {COLLECTION_NAME}")


def fill_database():
    check_model_argument()
    prepare_db()
    load_text_to_database()


def prepare_db():
    prepare_db_dir()
    prepare_db_client()
    prepare_db_collection()


def prepare_db_collection():
    global DB_COLLECTION
    global COLLECTION_NAME
    DB_COLLECTION = DB_CLIENT.get_or_create_collection(name=COLLECTION_NAME)


def load_text_to_database():
    if not args.dir:
        print("Не указан каталог с txt-файлами")
        exit(1)

    global TXT_DIR_PATH
    TXT_DIR_PATH = Path(args.dir)

    if not TXT_DIR_PATH.is_dir():
        print(f"'{TXT_DIR_PATH}' not exists or not a dir")
        exit(1)

    texts, ids, metadatas = get_text_data_from_folder()
    save_text_to_database_chunked(texts, ids, metadatas)


def save_text_to_database_chunked(
        texts: list[str], 
        ids: list[str], 
        metadatas: list[dict[str,str]], 
        batch_size: int = 10
):
    for i in range(0, len(texts), batch_size):
        slice_start = i
        slice_end = i+batch_size

        chink_of_texts = texts[slice_start:slice_end]
        chink_of_ids = ids[slice_start:slice_end]
        chink_of_metadatas = metadatas[slice_start:slice_end]
        
        chunk_of_vectors = text2vector(chink_of_texts)

        store_data_in_database(
            chunk_of_vectors, 
            chink_of_ids, 
            chink_of_metadatas, 
            chink_of_texts
        )


def store_data_in_database(
        vectors: list[list[float]], 
        ids: list[str], 
        metadatas: list[dict[str,str]], 
        texts: list[str]
):
    DB_COLLECTION.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas
    )
    

def get_text_data_from_folder() -> tuple[list[str], list[str], list[dict[str,str]]]:

    txt_files = list(TXT_DIR_PATH.rglob("*.txt"))

    if not txt_files:
        print(f"В каталоге '{TXT_DIR_PATH}' не найдено ни одного .txt файла.")
        exit(1)

    texts: list[str] = []
    ids: list[str] = []
    metadatas: list[dict] = []

    for file in txt_files:
        try:
            with open(file, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read().strip()

            if not content:
                print(f"Пропускаю пустой файл {file}")
                continue

            texts.append(content)

            file_id = str(file.relative_to(TXT_DIR_PATH)).replace('/', '_').replace('\\', '_')
            ids.append(file_id)
            metadatas.append({"source": str(file)})

        except Exception as e:
            print(f"Предупреждение: Не удалось прочитать файл {file.name}. Ошибка: {e}")

    return texts, ids, metadatas


def text2vector(input_text_or_texts) -> list[list[float]]:

    payload = {
        "model": MODEL_NAME,
        "input": input_text_or_texts
    }

    response = requests.post(EMB_URL, json=payload)
    response.raise_for_status()
    result = response.json()
    result_data = result["data"]

    # гарантируем порядок, совпадающий с порядком входных строк
    items = sorted(result_data, key=lambda x: x["index"])
    
    return [item["embedding"] for item in items]


def prepare_db_client():
    global DB_CLIENT
    DB_CLIENT = chromadb.PersistentClient(path=DB_PATH)
    pass


def prepare_db_dir():
    if not os.path.exists(DB_PATH):
        print(f'Создаю каталог "{DB_PATH}"')
        os.makedirs(DB_PATH)
    if not os.path.exists(DB_PATH):
        print(f"Каталог '{DB_PATH}' не создан!")
        exit(1)


def query_to_database(query_text: str):
    check_model_argument()
    prepare_db()
    query_to_database_real(query_text)


def query_to_database_real(query_text: str):
    query_vector = text2vector([query_text])[0]

    results = DB_COLLECTION.query(
        query_embeddings=query_vector,
        n_results=args.count
    )

    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        distance = results['distances'][0][i]
        # Красиво обрезаем длинный текст для вывода
        preview = doc[:200].replace('\n', ' ') + "..." if len(doc) > 200 else doc.replace('\n', ' ')
        print(f"[{i+1}] Файл: {meta['source']} (Дистанция: {distance:.4f})")
        print(f"    Текст: {preview}\n")


if args.list:
    list_models_on_server()
    exit(0)
elif args.dir:
    fill_database()
    exit(0)
elif args.query:
    query_to_database(args.query)
    exit(0)