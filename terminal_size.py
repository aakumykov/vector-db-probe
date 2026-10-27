import shutil

# Получаем размер терминала
terminal_size = shutil.get_terminal_size()

a = shutil.get_terminal_size().columns

# terminal_size - это именованный кортеж с атрибутами .columns и .lines
print(f"Ширина (колонок): {terminal_size.columns}")
print(f"Высота (строк):   {terminal_size.lines}")

# Также можно задать fallback (значение по умолчанию), если терминал не определен
size = shutil.get_terminal_size(fallback=(80, 24))

print(size)