from enum import Enum
import questionary

QUIT_MENU_CHOICE = ["Завершение работы"]

class Model(Enum):
    EMB_DOC_YANDEX_V2 = 'text-embeddings-v2-doc'
    EMB_QUERY_YANDEX_V2 = 'text-embeddings-v2-query'
    GPT_YANDEX_5_LIST = 'yandexgpt-5-lite'


def ask_for_model() -> str:
    model = questionary.select("Выберите модель для работы", [m.name for m in Model]+QUIT_MENU_CHOICE).ask()
    
    if QUIT_MENU_CHOICE == model:
        exit(0)
    else:
        print(f"Модель для работы '{model}'")
        return model

