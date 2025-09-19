"""домашка : собрать 20-30 текстов и присвоить каждому метку: спам или нормальное сообщение
в аудитории: прочитать данные csv , векторизировать тексты (CountVectorizer or TfidfVectorizer )
обучить классификатор (например LogisticRegression)
проверить работу на новых сообщениях """

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# открываем и читаем  csv файл
df = pd.read_csv("тексты.csv") 
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)
X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

pipe_nb = make_pipeline(CountVectorizer(), MultinomialNB())# две функции выполняются последовательно а также чтобы не вызывать два раза
pipe_nb.fit(X_train, y_train)# запустили преобразование всего массива
y_pred_nb = pipe_nb.predict(X_test)
print(" accuracy:", accuracy_score(y_test, y_pred_nb))

# Linear regression


pipe_lr = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=1000))
pipe_lr.fit(X_train, y_train)# обучаем логистич регрессию 
y_pred_lr = pipe_lr.predict(X_test)
print(" LR accuracy:", accuracy_score(y_test, y_pred_lr))


new_messages = [ 
    "Поздравляем! Вы победитель. Пришлите нам свой CVV код",
    "Добрый вечер! Встреча завтра в 12:00",
    "Получите компенсацию! Заполните форму прямо сейчас",
    "Презентация готова, можно начинать репетицию"
]

print("\n Предсказания для новых сообщений (LG):") # проверяем работу на новых сообщениях
for msg in new_messages:
    pred = pipe_lr.predict([msg])[0]
    label = "СПАМ" if pred == 1 else "НОРМ"
    print(f"'{msg}' {label}")