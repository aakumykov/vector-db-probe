
def read_file_from_home(file_name: str) -> str:
    from pathlib import Path
    return (Path.home() / file_name).read_text(encoding='utf-8').strip()