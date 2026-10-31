#!.venv/bin/python
import questionary

q = questionary

#c = questionary.text("Что?").ask()
c = q.text("Что?").ask()
print(f"Вот что: {c}")


