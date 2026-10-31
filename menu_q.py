#!.venv/bin/python
import questionary
q = questionary

def p(t: str):
	print(t)


#c = q.text("Что?").ask(); print(f"Вот что: {c}")

#pw = q.password("Введи-ка свой пароль!").ask(); p(f"Пароль (ха-ха): {pw}")

#some_coffee = q.confirm("Будешь кофе?").ask(); p(f"Андрюха будет кофе? - {some_coffee}")
#some_coffee = q.confirm("Будешь кофе?").ask(); p(f"Андрюха {if some_coffee: \"будет\" else \"не будет\"} кофе")

#color = q.select("Твой любимый цвет?", choices=["Синий","Красный","Голубой","Чирковый"]).ask(); p(color)

stuff = q.checkbox("Что взять в поход:", choices=["Грибы","Сыр","Плоскогубцы","Свисток","Тонкогубцы","Котёл для грешников","Спички","Ведро воды"]).ask(); p(f"В поход берём: {stuff}")
