import argparse
import os
import shutil
from pathlib import Path
from bs4 import BeautifulSoup

def clear_dir(folder_path):
    try:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path, exist_ok=True)
        print(f'Содержимое каталога {folder_path} успешно очищено.')
    except OSError as e:
        print(f'Ошибка при очистке {folder_path}: {e.strerror}')


def main():
    # 1. Настройка парсера аргументов командной строки
    parser = argparse.ArgumentParser(description="Извлечение текста из HTML файлов")
    parser.add_argument("html_dir", type=str, help="Путь к каталогу с HTML файлами")
    parser.add_argument("txt_dir", type=str, help="Путь к каталогу с TXT файлами")
    args = parser.parse_args()
    
    source_dir = Path(args.html_dir)
    target_dir = Path(args.txt_dir)
    
    # 2. Проверка существования каталога
    if not source_dir.is_dir():
        print(f"Ошибка: Каталог '{source_dir}' не найден или не является директорией.")
        return

    clear_dir(target_dir)

    # 3. Поиск всех файлов с расширениями .html и .htm
    html_files = list(source_dir.glob("*.{html}")) + list(source_dir.glob("*.htm")) + list(source_dir.glob("*.xhtml"))
    
    if not html_files:
        print("В указанном каталоге не найдено HTML файлов.")
        return

    print(f"Найдено файлов: {len(html_files)}. Начинаю обработку...")

    # 4. Обработка каждого файла
    for html_file in html_files:
        try:
            # Читаем HTML файл
            with open(html_file, 'r', encoding='utf-8', errors='replace') as file:
                html_content = file.read()
            
            # Парсим HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Удаляем ненужные теги (скрипты, стили, навигацию), чтобы не мусорить в тексте
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'noscript']):
                element.decompose()
            
            # Извлекаем текст. 
            # separator=' ' добавляет пробел между блоками, strip=True убирает лишние пробелы по краям
            text = soup.get_text(separator=' ', strip=True)
            
            # Формируем путь для нового .txt файла (меняем расширение)
            txt_file = target_dir / f"{html_file.stem}.txt"
            
            # Сохраняем текст
            with open(txt_file, 'w', encoding='utf-8') as file:
                file.write(text)
                
            print(f"Успешно: {html_file.name} -> {txt_file.name}")
            
        except Exception as e:
            print(f"Ошибка при обработке {html_file.name}: {e}")


if __name__ == "__main__":
    main()
