import pandas as pd
import numpy  as np
import os
import re
import logging
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout, SimpleRNN
from keras import utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import keras.backend.tensorflow_backend as tb


def preprocess_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
    text = re.sub(r'@[^\s]+', 'USER', text)
    text = re.sub(r'[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    return text.split(" ")


def get_sequences(tokenizer, x, len):
    sequences = tokenizer.texts_to_sequences(x)
    return pad_sequences(sequences, maxlen = len)

def learn_model_RNN (cfg, tokenizer, information = None):
    tb._SYMBOLIC_SCOPE.value = True
    if not information == None:
        information['text'] = 'Запуск обучающего модуля'
    print('Запуск обучающего модуля...')
    # считывание дата сэта
    n = ['id', 'date', 'name', 'text', 'typr', 'rep', 'rtw', 'faw', 'stcount', 'foll', 'frien', 'listcount']
    data_positive = pd.read_csv(os.getcwd() + cfg['positivePlace'], sep=';', error_bad_lines=False, names=n, usecols=['text'])
    data_negative = pd.read_csv(os.getcwd() + cfg['negativePlace'], sep=';', error_bad_lines=False, names=n, usecols=['text'])

    if not information == None:
        information['text'] = 'Датасэт считан'
    print('Датасэт считан')
    sample_size = min(data_positive.shape[0], data_negative.shape[0])

    labels = [1] * sample_size + [0] * sample_size

    raw_data = np.concatenate((data_positive['text'].values[:sample_size],
                            data_negative['text'].values[:sample_size]))

    data = [preprocess_text(t) for t in raw_data]

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)

    if not information == None:
        information['text'] = 'Датасэт подготовлен'
    print('Датасэт подготовлен')


    # Отображаем каждый текст в массив идентификаторов токенов
    x_train_seq = get_sequences(tokenizer, x_train, cfg['Sen_Lenght'])
    x_test_seq = get_sequences(tokenizer, x_test, cfg['Sen_Lenght'])

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = Sequential()
    model.add(Embedding(cfg['NUM'], 2, input_length=cfg['Sen_Lenght'],))
    model.add(SimpleRNN(cfg['RNNneurons']))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=cfg['optimizer'], 
                loss=cfg['lossType'], 
                metrics=cfg['metrics'])

    if not information == None:
        information['text'] = 'Модель скомпилирована, Идет обучение'
    print('Модель скомпилирована')

    model.fit  (x_train_seq, 
                y_train, 
                epochs=cfg['epoch'],
                batch_size=cfg['BatchSize'],
                validation_split=0.1)

    scores = model.evaluate(x_test_seq, y_test, verbose=1)
    
    if not information == None:
        information['text'] = 'Модель подготовлена'
    print('Модель подготовлена')
    
    accuracy = scores[1] * 100

    if not information == None:
        information['text'] = f'точность = {accuracy}'
    print('Точность = ', str(accuracy))
    
    exitName = f"model_{cfg['RNNneurons']}_{cfg['epoch']}_{'%i' % accuracy}.HDF5"
    
    model.save(f"{os.getcwd()}{cfg['savePath']}\\{exitName}")
    
    if not information == None:
        information['text'] = f'Модель {exitName} сохранена'
    print(f"Модель {exitName} Сохранена")
    return True
