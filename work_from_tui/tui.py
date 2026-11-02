#!.venv/bin/python
from ask_for_action import ask_for_action
from ask_for_action import Action
from encode_text import encode_text
from query_db import query_db
from ask_model import ask_model

action = ask_for_action()


if Action.ENCODE_TEXT == action:
    encode_text()
elif Action.QUERY_DB == action:
    query_db()
elif Action.ASK_MODEL == action:
    ask_model()
elif Action.QUIT == action:
    exit(0)
else:
    print(f"Неизвестное действие '{action}'")