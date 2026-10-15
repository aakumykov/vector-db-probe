from collections.abc import Callable
from pathlib import Path


def for_text_in_files(folder_path: str, block: Callable[[str,str], any]):
    for p in Path(folder_path).iterdir():
        if p.is_file() and p.suffix == '.txt':
            text = p.read_text(encoding='utf-8')
            block(text, p.name)


def process(file_name: str, text: str):
    print(f'{file_name}: {text}')


#for_text_in_files("txt", lambda s: print(f'ТЕКСТ: {s}'))
# for_text_in_files("txt", process)

