from pick_directory_tui import pick_directory_tui
import os
from ask_for_model import ask_for_model
import chromadb
from chromadb.api import ClientAPI
import openai
from read_file_from_home import read_file_from_home

def encode_text(db_path="db"):

    # model = ask_for_model()
    selected_dir = pick_directory_tui("../txt")
    print(f"Выбранный каталог: {selected_dir}")

    file_paths = list_dir(selected_dir)
    print(f"=== Текстовые файлы в каталоге '{selected_dir}': ===")
    # for f_path in file_paths: print(f_path)

    selected_model = ask_for_model()
    print(f"Выбранная модель: {selected_model}")

    # db = chromadb.PersistentClient(path=db_path)

    for txt_path in file_paths:
        file_contents = get_file_contents(txt_path)
        vector = vectosize_text(file_contents, selected_model)


def get_file_contents(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read().strip()


def list_dir(directory_path: str) -> list[str]:
    return [ f.path for f in os.scandir(directory_path) if f.is_file() and f.name.lower().endswith('.txt') ]


def vectosize_text(
        text: str, 
        model_name: str
) -> list[float]:
    YANDEX_API_KEY = read_file_from_home('.yandex_ai_token')
    