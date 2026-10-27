#!.venv/bin/python

import re
from argparse import ArgumentError
from typing import Any
import shutil

import requests
import argparse
import os
import sys
from enum import Enum
from os.path import expanduser
from pathlib import Path

import chromadb
from chromadb import ClientAPI

from langchain_text_splitters import RecursiveCharacterTextSplitter


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
DB_CLIENT: ClientAPI
DB_COLLECTION: Any

MODEL_NAME: str

TXT_DIR_PATH: Path

# Константы
KEY_TEXT = "TEXT"
KEY_FILE_NAME = "FILE_NAME"
KEY_ID = "ID"

DEFAULT_TEXT_CHUNK_SIZE = 8192
DEFAULT_BATCH_SIZE = 10 # Размер пакета для обработки моделью

parser = argparse.ArgumentParser(description="Создание векторной БД из txt файлов и семантический поиск.")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-lm", "--list-models", action='store_true', dest="list_models", help="Вывести список доступных моделей")
group.add_argument("-lc", "--list-collections", action='store_true', dest="list_collections", help="Вывести список коллекций из БД")
group.add_argument("--load", dest="dir", type=str, help="Путь к каталогу с txt файлами")
group.add_argument("-q", "--query", type=str, help="Текст запроса к БД")
group.add_argument("-d", "--delete", type=str, help="Удалить коллекцию (вместе с данными) из БД")
group.add_argument("-e", "--encode", type=str, help="Закодировать строку в вектор")

parser.add_argument("-m", "--model", type=str, required=False, help="Название модели")
parser.add_argument("-c", "--count", type=int, required=False, default=DEFAULT_RESULTS_COUNT, help=f"Количество результатов поиска (по умолчанию: {DEFAULT_RESULTS_COUNT})")
args = parser.parse_args()


def log_d(text: str):
    print(text)


def log_w(text: str):
    print(f"ПРЕДУПРЕЖДЕНИЕ: {text}")


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

    texts_with_data = get_texts_from_folder(DEFAULT_TEXT_CHUNK_SIZE)
    
    save_text_to_database_chunked(texts_with_data)


def save_text_to_database_chunked(
        texts_with_data: list[dict[str,str]],
        batch_size: int = DEFAULT_BATCH_SIZE
):
    for i in range(0, len(texts_with_data), batch_size):
        slice_start = i
        slice_end = i+batch_size

        data_chunk = texts_with_data[slice_start:slice_end]
        
        chunk_of_texts = list(map(lambda item: item[KEY_TEXT], data_chunk))
        chunk_of_ids = list(map(lambda item: item[KEY_ID], data_chunk))
        chunk_of_metadatas = list(map(lambda item: {KEY_FILE_NAME: item[KEY_FILE_NAME]}, data_chunk))

        for i, text in enumerate(chunk_of_texts):
            print(f"  сохраняю текст [{len(text)} симв., {chunk_of_metadatas[i][KEY_FILE_NAME]}] '{text[:50]}'")

        chunk_of_vectors = text_list_2_vectors_list(chunk_of_texts)

        store_data_in_database(
            chunk_of_vectors, 
            chunk_of_ids, 
            chunk_of_metadatas, 
            chunk_of_texts
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
    

def get_texts_from_folder(chunk_max_size: int = DEFAULT_TEXT_CHUNK_SIZE) -> list[dict[str,str]]:
    """
    Returns:
        [ { "file_text: "string", "file_name": "string", "id": "string" } ] 
    """

    txt_files = list(TXT_DIR_PATH.rglob("*.txt"))

    if not txt_files:
        print(f"В каталоге '{TXT_DIR_PATH}' не найдено ни одного .txt файла.")
        exit(1)

    result = []

    for file in txt_files:
        f_name = file.name
        try:
            with open(file, 'r', encoding='utf-8', errors='replace') as f:
                text_content = f.read().strip()

            if not text_content:
                print(f"Пропускаю пустой файл {file}")
                continue

            print(f"Читаю файл '{f_name}'")

            c = 0
            for text_chunk in split_text_to_chunks(text_content, chunk_max_size=chunk_max_size):
                chunk_id = f"{f_name}_{c}"
                text_with_data = {
                    KEY_TEXT: text_chunk,
                    KEY_FILE_NAME: file.name,
                    KEY_ID: chunk_id
                }
                result.append(text_with_data)
                c+=1

        except Exception as e:
            print(f"Предупреждение: Не удалось прочитать файл {f_name}. Ошибка: {e}")

    return result


def encode_string(text: str):
    check_model_argument()
    vec = text2vector(text)
    print(f"Текст ({len(text)} символов): {text}")
    print(f"Вектор ({len(vec)} значений):")
    print(vec)


def text2vector(input_text: str) -> list[float]:

    payload = {
        "model": MODEL_NAME,
        "input": input_text
    }

    response = requests.post(
        EMB_URL, 
        json=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    response.raise_for_status()
    result = response.json()
    result_data = result["data"][0]["embedding"]
            
    return result_data


def text_list_2_vectors_list(text_list: list[str]) -> list[list[float]]:

    payload = {
        "model": MODEL_NAME,
        "input": text_list
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

    results = DB_COLLECTION.query(
        query_embeddings=text2vector(query_text),
        n_results=args.count
    )

    zipped_data = zip(
        results['documents'][0], 
        results['metadatas'][0], 
        results['distances'][0]
    )

    for i, (doc, meta, distance) in enumerate(zipped_data):
        distance = results['distances'][0][i]
        preview = wrap_string_to_width(doc)
        print(f"[{i+1}] Файл: {meta[KEY_FILE_NAME]} (Дистанция: {distance:.4f})")
        print(f"    Текст: {preview}\n")


def wrap_string_to_width(text: str, width: int = None, delimiter: str = "\n") -> list[str]:
    if not width:
        actual_width = shutil.get_terminal_size().columns
    else:
        actual_width = width

    return delimiter.join(split_text_to_chunks(text, actual_width, 0))


def delete_collection():
    prepare_db_client()

    name = args.delete
    log_d(f"Удаление коллекции {name}")
    
    global DB_CLIENT
    if DB_CLIENT.list_collections().count(name) > 0:
        DB_CLIENT.delete_collection(name)
        log_d(f"Коллекция '{name}' удалена")
    else:
        log_w(f"Нет коллекции '{name}'")


def list_collections():
    prepare_db_client()

    print("=== Коллекции в БД ===")
    for coll in DB_CLIENT.list_collections():
        print(coll.name)


def split_text_to_chunks(
        input_text: str, 
        chunk_max_size: int = 512, 
        chunk_overlap_percent: int = 30
) -> list[str]:

    overlap_len = round(chunk_overlap_percent / 100.0 * chunk_max_size)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_max_size,
        chunk_overlap=overlap_len,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False
    )

    return text_splitter.split_text(input_text)


def split_long_text_in_list(text_list: list[str]) -> list[str]:

    chunk_list: list[str] = []

    for i, text in enumerate(text_list):
        for chunk in split_text_to_chunks(text):
            chunk_list.append(chunk)
            
    return chunk_list



if args.list_models:
    list_models_on_server()
    exit(0)

elif args.list_collections:
    list_collections()
    exit(0)

elif args.dir:
    fill_database()
    exit(0)

elif args.query:
    query_to_database(args.query)
    exit(0)

elif args.delete:
    delete_collection()
    exit(0)

elif args.encode:
    encode_string(args.encode)
    exit(0)

else:
    raise Exception(f"Неизвестный режим работы {sys.argv}")