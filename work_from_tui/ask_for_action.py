from enum import Enum
import questionary

class Action(Enum):
    ENCODE_TEXT = "Закодировать текст"
    QUERY_DB = "Найти в БД"
    ASK_MODEL = "Спросить модель"


