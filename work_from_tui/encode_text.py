from pick_directory_tui import pick_directory_tui
from pathlib import Path
import os


def encode_text():
    # model = ask_for_model()
    selected_dir = pick_directory_tui("../txt")
    print(f"Выбранный каталог: {selected_dir}")

    file_paths = list_dir(selected_dir)
    print(f"=== Текстовые файлы в каталоге '{selected_dir}': ===")
    for f_path in file_paths:
        print(f_path)


def list_dir(directory_path: str) -> list[str]:
    return [ f.path for f in os.scandir(directory_path) if f.is_file() and f.name.lower().endswith('.txt') ]

