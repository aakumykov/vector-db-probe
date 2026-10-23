#!.venv/bin/python

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


parser = argparse.ArgumentParser(description="Создание векторной БД из txt файлов и семантический поиск.")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-l", "--list", action='store_true', help="Показатб список доступных моделей")
group.add_argument("-f", "--fill", type=str, dest="source_directory", help="Путь к каталогу с txt файлами")
group.add_argument("-q", "--query", type=str, help="Текст запроса к БД")

parser.add_argument("-m", "--model", type=str, dest="model_name", help="Название модели")
parser.add_argument("-c", "--count", type=int, required=False, default=DEFAULT_RESULTS_COUNT, help=f"Количество результатов поиска (по умолчанию: {DEFAULT_RESULTS_COUNT})")
args = parser.parse_args()


def list_models_on_server() -> list[str]:
    url = f"http://{SERVER_HOST}:{SERVER_PORT}{MODELS_PATH}"
    response = requests.get(url)
    response.raise_for_status()
    model_ids = [model["id"] for model in response.json()["data"]]
    return model_ids


if args.list:
    print(f"===== Models on {SERVER_HOST}: =====")
    for m in list_models_on_server():
        print(m)

