import pandas as pd
import numpy as np
import nltk
import re
import string
import pymorphy3
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = FastAPI()

nltk.download('punkt_tab')

with open('best_model_250_films.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer_250_films.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

def fun_punctuation_text(text: str) -> str:
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    text = ''.join([i if not i.isdigit() else '' for i in text])
    text = ''.join([i if i.isalpha() else ' ' for i in text])
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub('[a-z]', '', text, flags=re.I)
    st = '❯\xa0'
    text = ''.join([ch if ch not in st else ' ' for ch in text])
    return text

def fun_lemmatizing_text(text: str) -> str:
    tokens = word_tokenize(text)
    res = list()
    for word in tokens:
        p = pymorphy3.MorphAnalyzer(lang='ru').parse(word)[0]
        res.append(p.normal_form)

    text = ' '.join(res)
    
    return text

def fun_tokenize(text: str) -> str:
    russian_stopwords = stopwords.words("russian") 
    russian_stopwords.extend(['т.д.', 'т', 'д', 'это','который','с','своём','всем','наш', 'свой', 'также', 'которые', '–', 'очень', 'нужно', 'просто', 'например', 'всё', 'поэтому', 'который', 'какие', 'такой', 'другой', 'каждый', 'свой', 'должный', 'быть', 'тот', 'сам', 'свой', 'мой', 'тот', 'кто', 'ваш', 'мы', 'какой', 'простой', 'либо', 'самый', 'ещё', 'любой', 'несколько', 'некоторый', 'должный', 'являться', 'новый', 'свой', 'этот', 'разный', 'самый'])
    t = word_tokenize(text)
    tokens = [token for token in t if token not in russian_stopwords]
    text = ' '.join(tokens)
    return text

def fun_pred_text(text: str) -> str:
    text = fun_punctuation_text(text)
    text = fun_lemmatizing_text(text)
    text = fun_tokenize(text)
    return text

def predict_cluster(text: str) -> tuple:
    text_vectorizer = vectorizer.transform([fun_pred_text(text)])
    prediction = model.predict(text_vectorizer)
    probabilities = model.predict_proba(text_vectorizer)
    rez1 = f'Класс: {prediction[0]}'
    rez2 = f'Вероятности: {probabilities}'
    mapping = {
        0 : ' 0 - кластер',
        1 : ' 1 - кластер',
        2 : ' 2 - кластер',
        3 : ' 3 - кластер',
        4 : ' 4 - кластер',
        5 : ' 5 - кластер',
        6 : ' 6 - кластер',
    }

    selected_cluster = mapping[prediction[0]]
    return selected_cluster, rez2

# Мщдель для входных данных
class Item(BaseModel):
    text: str

# Метод для подключения к интерфейсу
@app.post('/predict')
def post_pred_text(item: Item):
    return {'cluster' : predict_cluster(item.text)}
