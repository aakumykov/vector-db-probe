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
from chromadb import ClientAPI
from huggingface_hub import login
from sentence_transformers import SentenceTransformer

SERVER_HOST="localhost"
SERVER_PORT=9931
QUERY_PATH="/v1/embeddings"
MODELS_PATH="/v1/models"
DEFAULT_RESULTS_COUNT=5

DB_PATH="db"
DB_CLIENT: ClientAPI
MODEL_NAME: str
COLLECTION_NAME: str
TXT_DIR_PATH: Path

parser = argparse.ArgumentParser(description="Создание векторной БД из txt файлов и семантический поиск.")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-l", "--list", action='store_true', help="Вывести список доступных моделей")
group.add_argument("-f", "--fill", action='store_true', help="Путь к каталогу с txt файлами")
group.add_argument("-q", "--query", type=str, help="Текст запроса к БД")

parser.add_argument("-d", "--dir", type=str, required=False, help="Каталог с txt-файлами (используется при наполнении БД)")
parser.add_argument("-m", "--model", type=str, required=False, help="Название модели")
parser.add_argument("-c", "--count", type=int, required=False, default=DEFAULT_RESULTS_COUNT, help=f"Количество результатов поиска (по умолчанию: {DEFAULT_RESULTS_COUNT})")
args = parser.parse_args()


def log_d(text: str):
    print(text)

def list_models_on_server():
    url = f"http://{SERVER_HOST}:{SERVER_PORT}{MODELS_PATH}"
    response = requests.get(url)
    response.raise_for_status()
    model_ids = [model["id"] for model in response.json()["data"]]
    print(f"===== Models on {SERVER_HOST}: =====")
    for m in model_ids:
        print(m)


def fill_database():
    if not args.model:
        print("He yka3aHo umR moDejlu")
        exit(1)

    global MODEL_NAME
    MODEL_NAME = args.model
    log_d(f"MODEL_NAME: {MODEL_NAME}")

    global COLLECTION_NAME
    COLLECTION_NAME = re.sub("[^0-9a-zA-Z]+", "_", MODEL_NAME)
    log_d(f"COLLECTION_NAME: {COLLECTION_NAME}")

    fill_db_real()


def fill_db_real():
    prepare_db_dir()
    prepare_db_client()
    load_text_to_database()


def load_text_to_database():
    if not args.dir:
        print("Не указан каталог с txt-файлами")
        exit(1)

    global TXT_DIR_PATH
    TXT_DIR_PATH = Path(args.dir)

    if not TXT_DIR_PATH.is_dir():
        print(f"'{TXT_DIR_PATH}' not exists or not a dir")
        exit(1)

    load_text_to_database_real()


def load_text_to_database_real():

    txt_files = list(TXT_DIR_PATH.rglob("*.txt"))

    if not txt_files:
        print(f"В каталоге '{TXT_DIR_PATH}' не найдено ни одного .txt файла.")
        exit(1)

    texts = []
    ids = []
    metadatas = []

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


def prepare_db_client():
    global DB_CLIENT
    DB_CLIENT = chromadb.PersistentClient(path=DB_PATH)


def prepare_db_dir():
    if not os.path.exists(DB_PATH):
        print(f'Создаю каталог "{DB_PATH}"')
        os.makedirs(DB_PATH)
    if not os.path.exists(DB_PATH):
        print(f"Каталог '{DB_PATH}' не создан!")
        exit(1)


if args.list:
    list_models_on_server()
    exit(0)
elif args.fill:
    fill_database()
    exit(0)