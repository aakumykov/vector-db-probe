#!.venv/bin/python

import argparse
import os
import sys
from enum import Enum
from os.path import expanduser
from pathlib import Path

import chromadb
from huggingface_hub import login
from sentence_transformers import SentenceTransformer

# Запрещаем любые сетевые запросы к Hugging Face
os.environ["HF_HUB_OFFLINE"] = "1" 
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class Models(Enum):
    INTFLOAT_MULTILINGUAL_E5_SMALL = "intfloat/multilingual-e5-small"
    COINTEGRATED_RUBERT_TINY = "cointegrated/rubert-tiny2"
    SBERT_LARGE_NLU_RU = "ai-forever/sbert_large_nlu_ru"
    # FRIDA = "ai-forever/FRIDA"
    # RU_EN_ROSBERTA = "ai-forever/ru-en-RoSBERTa" # Нужны спец. префиксы
    # DEEPVK_USER_BASE = "deepvk/USER-base" # Нужны спец. префиксы
    # "sergeyzh/LaBSE-ru-turbo"
    # "ai-forever/sbert_large_nlu_ru"


def list_available_models():
    print("Доступныя модели:")
    for item in Models:
        print(f'* "{item.value}", код: {item.name}')


def hf_auth_token(token_path=f'{expanduser("~")}/.hf_token'):
    token = Path(token_path).read_text()
    if not token:
        raise ValueError(f"No or empty token from '{token_path}'")
    return token



def auth_in_hugging_face(token):
    login(token)


def load_model(name):
    print(f"Загрузка модели {name}...")
    return SentenceTransformer(name)



def load_texts_from_directory(directory_path):
    """Читает все .txt файлы из указанного каталога."""
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        raise ValueError(f"Ошибка: '{directory_path}' не является существующим каталогом.")
    
    texts = []
    ids = []
    metadatas = []
    
    # Ищем все файлы с расширением .txt
    txt_files = list(dir_path.rglob("*.txt")) # rglob ищет и во вложенных папках
    
    if not txt_files:
        raise ValueError(f"В каталоге '{directory_path}' не найдено ни одного .txt файла.")
        
    print(f"Найдено {len(txt_files)} текстовых файлов. Читаю содержимое...")
    
    for file in txt_files:
        try:
            with open(file, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read().strip()
                
            if not content:
                print(f"Пропускаю пустой файл {file}")
                continue  # Пропускаем пустые файлы
                
            texts.append(content)
            # Используем относительный путь как уникальный ID (заменяем слеши, чтобы не было проблем)
            file_id = str(file.relative_to(dir_path)).replace('/', '_').replace('\\', '_')
            ids.append(file_id)
            metadatas.append({"source": str(file)})
            
        except Exception as e:
            print(f"Предупреждение: Не удалось прочитать файл {file.name}. Ошибка: {e}")
            
    return texts, ids, metadatas



def convert_text_to_embeddings(model, directory, collection):

    texts, ids, metadatas = load_texts_from_directory(directory)
    print(f"Успешно загружено {len(texts)} документов.")
    
    print("Создание эмбеддингов (это может занять время)...")
    embeddings = model.encode(texts).tolist()

    if not embeddings:
        raise Exception("Нет (пустой список) эмбеддингов")

    # Используем upsert вместо add, чтобы при повторном запуске скрипта
    # файлы просто обновились, а не вызвали ошибку дублирования ID
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    print(f"Данные успешно сохранены в коллекцию '{args.collection}'. Всего документов: {collection.count()}")



def prepare(m_name: str, c_name: str, path_to_db: str):
    import os
    if not os.path.exists(path_to_db):
        os.makedirs(path_to_db)

    client = chromadb.PersistentClient(path=path_to_db)
    return load_model(m_name), client.get_or_create_collection(name=c_name)



def query_database(m_name, collection):
    while True:
        query = input("\nВаш запрос: ").strip()
        if query.lower() in ['exit', 'quit', 'выход', 'q']:
            break
        if not query:
            continue
            
        query_embedding = m_name.encode([query]).tolist()
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=args.top_k
        )
        
        print(f"\n--- Топ-{args.top_k} результатов ---")
        for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            distance = results['distances'][0][i]
            # Красиво обрезаем длинный текст для вывода
            preview = doc[:200].replace('\n', ' ') + "..." if len(doc) > 200 else doc.replace('\n', ' ')
            print(f"[{i+1}] Файл: {meta['source']} (Дистанция: {distance:.4f})")
            print(f"    Текст: {preview}\n")



    
parser = argparse.ArgumentParser(description="Создание векторной БД из txt файлов и семантический поиск.")
group = parser.add_mutually_exclusive_group(required=True)  # required=True делает один из аргументов обязательным
group.add_argument("--list", action='store_true', help="Показатб список доступных моделей")
group.add_argument("--load-from", type=str, dest="source_directory", help="Путь к каталогу с txt файлами")
group.add_argument("--query", action='store_true', help="Режим запросов к БД")

parser.add_argument("--model", type=str, dest="model_code", help="Код модели")
parser.add_argument("-c", "--collection", type=str, default="my_txt_db", help="Имя коллекции в БД (по умолчанию: my_txt_db)")
parser.add_argument("-k", "--top_k", type=int, default=5, help="Количество результатов поиска (по умолчанию: 5)")
args = parser.parse_args()


if args.list:
    list_available_models()
    sys.exit(0)


DB_DIR="db"

model_enum = Models[args.model_code]
model_name = model_enum.value
collection_name = model_enum.name
db_name = collection_name.lower()
db_path = os.path.join(DB_DIR, db_name)


if args.source_directory:
    auth_in_hugging_face(hf_auth_token())
    m, c = prepare(model_name, collection_name, db_path)
    convert_text_to_embeddings(m, args.source_directory, c)
elif args.query:
    auth_in_hugging_face(hf_auth_token())
    m, c = prepare(model_name, collection_name, db_path)
    query_database(m, c)
else:
    raise Exception("Неизвестно, что делать (аргументы)")


