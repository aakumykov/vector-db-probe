from enum import Enum
import questionary
q = questionary

class Colors(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    TEAL = "чирковый";

one_more_color = "чёрный"

color_list = [c.value for c in Colors] + [one_more_color]

# for i,item in enumerate(color_list):
#     print(f"{i}) {item}")

choice = q.select("Какой цвет?", [c.value for c in Colors]+[one_more_color]).ask()
