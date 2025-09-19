import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer  # можно заменить на TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Загрузка данных
df = pd.read_csv("тексты.csv")  # файл должен содержать столбцы: message, label
X = df["message"]
y = df["label"]  # 1 — СПАМ, 0 — НОРМ

# 2. Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 3. Создание и обучение пайплайна
pipe = make_pipeline(
    CountVectorizer(),  # можно заменить на TfidfVectorizer()
    LogisticRegression(max_iter=1000)
)

pipe.fit(X_train, y_train)

# 4. Оценка точности
y_pred = pipe.predict(X_test)
print(f"📊 Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# 5. Проверка на новых сообщениях
new_messages = [
    "Поздравляем! Вы победитель. Пришлите нам свой CVV код",
    "Добрый вечер! Встреча завтра в 12:00",
    "Получите компенсацию! Заполните форму прямо сейчас",
    "Презентация готова, можно начинать репетицию"
]

print("\n📬 Предсказания для новых сообщений:")
for msg in new_messages:
    pred = pipe.predict([msg])[0]
    label = "СПАМ" if pred == 1 else "НОРМ"
    print(f"'{msg}' → {label}")
