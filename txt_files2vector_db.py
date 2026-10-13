import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb


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


def load_model(name):
    print(f"Загрузка модели {name}...")
    model = SentenceTransformer(name)


def convert_text_to_embeddings(directory, collection):

    texts, ids, metadatas = load_texts_from_directory(directory)
    print(f"Успешно загружено {len(texts)} документов.")
    
    print("Создание эмбеддингов (это может занять время)...")
    embeddings = model.encode(texts).tolist()

    if not embeddings:
        raise StateError("Нет (пустой список) эмбеддингов")

    # Используем upsert вместо add, чтобы при повторном запуске скрипта
    # файлы просто обновились, а не вызвали ошибку дублирования ID
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    print(f"Данные успешно сохранены в коллекцию '{args.collection}'. Всего документов: {collection.count()}")



def prepare(collection_name, db_path="./vector_db_storage"):
    # model = load_model("cointegrated/rubert-tiny2")
    model = load_model("intfloat/multilingual-e5-small")
    
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)

    return model, collection



def query_database(model, collection):
    while True:
        query = input("\nВаш запрос: ").strip()
        if query.lower() in ['exit', 'quit', 'выход', 'q']:
            break
        if not query:
            continue
            
        query_embedding = model.encode([query]).tolist()
        
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
group.add_argument("--load-from", type=str, dest="directory", help="Путь к каталогу с txt файлами")
group.add_argument("--query", action='store_true', help="Режим запросов к БД")
parser.add_argument("-c", "--collection", type=str, default="my_txt_db", help="Имя коллекции в БД (по умолчанию: my_txt_db)")
parser.add_argument("-k", "--top_k", type=int, default=3, help="Количество результатов поиска (по умолчанию: 3)")
args = parser.parse_args()


if args.directory:
    model, collection = prepare(collection_name=args.collection)
    convert_text_to_embeddings(args.directory, collection)
elif args.query:
    model, collection = prepare(collection_name=args.collection)
    query_database(model, collection)
else:
    raise Exception("Неизвестно, что делать (аргументы)")


