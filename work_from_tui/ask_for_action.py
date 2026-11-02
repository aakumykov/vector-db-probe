from enum import Enum
import questionary
import ask_model, encode_text, query_db
from menu_common import QUIT_MENU_CHOICE

class Action(Enum):
    ENCODE_TEXT = "Кодировать текст в БД"
    QUERY_DB = "Искать в БД"
    ASK_MODEL = "Спрашивать модель"
    QUIT = QUIT_MENU_CHOICE


def ask_for_action() -> Action:

    action_text = questionary.select(
        "Что будем делать?",
        use_arrow_keys=True,
        use_shortcuts=True,
        use_jk_keys=True,
        choices = [a.value for a in Action]
    ).ask()


    return Action(action_text)
