#!.venv/bin/python

import questionary

action = questionary.select(
    "Что делаем?",
    choices=["Запустить задачу А", "Запустить задачу Б", "Выход"]
).ask()
