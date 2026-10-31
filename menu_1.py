#!.venv/bin/python

import os
import time

def main():
    while True:
        # Очищаем консоль (для Windows 'cls', для Linux/Mac 'clear')
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 30)
        print("📋 ГЛАВНОЕ МЕНЮ")
        print("=" * 30)
        print("1. 🚀 Запустить задачу А")
        print("2. 🛠  Запустить задачу Б")
        print("3. 🧹 Очистить логи")
        print("0. 🚪 Выход")
        print("=" * 30)
        
        choice = input("Введите номер команды: ").strip()
        
        if choice == '1':
            print("\n>> Выполняю задачу А...")
            time.sleep(1) # Имитация работы
        elif choice == '2':
            print("\n>> Выполняю задачу Б...")
            time.sleep(1)
        elif choice == '3':
            print("\n>> Логи очищены!")
            time.sleep(1)
        elif choice == '0':
            print("До свидания!")
            break
        else:
            print("\n❌ Неверный ввод. Попробуйте еще раз.")
            time.sleep(1)

if __name__ == "__main__":
    main()
