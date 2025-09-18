"""домашка : собрать 20-30 текстов и присвоить каждому метку: спам или нормальное сообщение
в аудитории: прочитать данные csv , векторизировать тексты (CountVectorizer or TfidfVectorizer )
обучить классификатор (например LogisticRegression)
проверить работу на новых сообщениях """

# классификация 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

text = ["купи афон за 100 рублей", #spam
"встречаа в офисе в 10.00", #not spam
"вы выиграли миллион,пришлите денег", #spam
"Отчет: продажи за квартал 2025",#not spam
"Срочно! Позвони сейчас и получи приз",#spam
"Напоминаем про оплату счета"]#not spam"

lable = [1,0,1,0,1,0] #1- spam; 0 - not spam

text_train,text_test,y_train,y_test = train_test_split(text,lable,test_size=0.33,random_state=42)

pipe = make_pipeline(# две функции выполняются последовательно а также чтобы не вызывать два раза
    CountVectorizer(),# задача сделать мешок слов , подсчет вхождения ... на выходе получается матрица
    MultinomialNB()
)

pipe.fit(text_train,y_train)# запустили преобразование всего массива
y_pred = pipe.predict(text_test)
print(y_pred)
print(f"Accurancy:{accuracy_score(y_test,y_pred)}")

print("New: ",pipe.predict(["Поздравляем !Вы победитель . пришлит нам свой CVV код"])[0])
print("New: ",pipe.predict(["Добрый вечер!Встреча завтра в"])[0])

# Linear regression

text_array = np.array(text)  #преобразовали в массив array
x = np.arange(len(text)).reshape(-1,1) 
y = text_array

model = LogisticRegression()
model.fit(x,y) # обучаем логистич регрессию 

report = np.array([[len(text_array)]])
logistic_pred = model.predict(report)[0]
print(f" Logistic regression prediction next week:{logistic_pred}")


