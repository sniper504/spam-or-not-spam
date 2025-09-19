import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv("тексты.csv")  # Убедись, что файл сохранён в UTF-8
X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

pipe_nb = make_pipeline(CountVectorizer(), MultinomialNB())
pipe_nb.fit(X_train, y_train)
y_pred_nb = pipe_nb.predict(X_test)
print(" accuracy:", accuracy_score(y_test, y_pred_nb))


pipe_lr = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=1000))
pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_test)
print(" LR accuracy:", accuracy_score(y_test, y_pred_lr))


new_messages = [
    "Поздравляем! Вы победитель. Пришлите нам свой CVV код",
    "Добрый вечер! Встреча завтра в 12:00",
    "Получите компенсацию! Заполните форму прямо сейчас",
    "Презентация готова, можно начинать репетицию"
]

print("\n Предсказания для новых сообщений (Logistic Regression):")
for msg in new_messages:
    pred = pipe_lr.predict([msg])[0]
    label = "СПАМ" if pred == 1 else "НОРМ"
    print(f"'{msg}' {label}")
