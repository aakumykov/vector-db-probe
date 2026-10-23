#!.venv/bin/python
from argparse import ArgumentError

import requests
import argparse
import os
import sys
from enum import Enum
from os.path import expanduser
from pathlib import Path

import chromadb
from huggingface_hub import login
from sentence_transformers import SentenceTransformer

SERVER_HOST="localhost"
SERVER_PORT=9931
QUERY_PATH="/v1/embeddings"
MODELS_PATH="/v1/models"
DEFAULT_RESULTS_COUNT=5

DB_PATH="db"


parser = argparse.ArgumentParser(description="Создание векторной БД из txt файлов и семантический поиск.")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-l", "--list", action='store_true', help="Вывести список доступных моделей")
group.add_argument("-f", "--fill", action='store_true', help="Путь к каталогу с txt файлами")
group.add_argument("-q", "--query", type=str, help="Текст запроса к БД")

parser.add_argument("-d", "--dir", type=str, required=False, help="Каталог с txt-файлами (используется при наполнении БД)")
parser.add_argument("-m", "--model", type=str, required=False, help="Название модели")
parser.add_argument("-c", "--count", type=int, required=False, default=DEFAULT_RESULTS_COUNT, help=f"Количество результатов поиска (по умолчанию: {DEFAULT_RESULTS_COUNT})")
args = parser.parse_args()


def list_models_on_server():
    url = f"http://{SERVER_HOST}:{SERVER_PORT}{MODELS_PATH}"
    response = requests.get(url)
    response.raise_for_status()
    model_ids = [model["id"] for model in response.json()["data"]]
    print(f"===== Models on {SERVER_HOST}: =====")
    for m in model_ids:
        print(m)


def fill_database(model_name: str):
    if not args.dir:
        print("Не указан каталог с txt-файлами")
        exit(1)
    fill_db_real()



def fill_db_real():
    prepare_db_dir()
    prepare_db_client()


def prepare_db_client():
    pass


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
    fill_database(args.model)
    exit(0)