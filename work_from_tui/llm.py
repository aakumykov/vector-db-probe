#!.venv/bin/python
import questionary
from enum import Enum
import openai
from pathlib import Path
from work_from_tui.ask_for_model import ask_for_model
from read_file_from_home import read_file_from_home

model = ask_for_model()

action = ask_for_action()

#menu = q.select("Главное меню", [
#    "Закодировать тексты из каталога",
#]).ask()
