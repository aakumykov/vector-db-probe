from collections.abc import Callable
from pathlib import Path

def for_text_in_files(folder_path: str, block: Callable[[str], any]):
   texts = [p.read_text(encoding='utf-8') for p in Path(folder_path).iterdir() if p.is_file() and p.suffix == '.txt']
   for t in texts:
       block(t)

def process(s: str):
    print(f'ПИЗДА: {s}')

#for_text_in_files("txt", lambda s: print(f'ТЕКСТ: {s}'))
for_text_in_files("txt", process)

