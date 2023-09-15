import time
import os
import re
import traceback
import subprocess
import configparser
from threading import Thread 
import math


StandartCfg = {
    'epoch'             : 5,                                    # Количество эпох обучения
    'NUM'               : 100000,                               # Размер словаря
    'Sen_Lenght'        : 28,                                   # Высота матрицы (максимальное количество слов в твите)
    'RNNneurons'        : 16,                                   # Количество нейронов в рекурентной 
    'BatchSize'         : 128,                                  # Размер батча
    'lossType'          : 'binary_crossentropy',                # binary_crossentropy|binary_crossentropy  Тип расчета ошибки
    'optimizer'         : 'adam',                               # adam|adam Оптимизатор
    'metrics'           : ['accuracy'],                         # Метрика оценки точности сети
    'positivePlace'     : '/DataSet/positive.csv',              # Местоположение позитивного датасэта
    'negativePlace'     : '/DataSet/negative.csv',              # Местоположение негативного датасэта
    'name'              : 'izumrud',                            # Имя программы
    'savePath'          : '/Models',                            # Папка в которую сохраняются модели              
    'version'           : '1.0.0',                              # Версия программы
    'modelPlace'        : '/Models/Model_20_16_72m.HDF5',       # Местоположение обученной модели
    'vecsumMax'         : 0.6,                                  # Верхний предел суммы векторов выше которого текст считается положительным
    'vecsumMin'         : 0.4,                                  # Нижний предел суммы векторов ниже которого текст считается отрицательным
    'autoInstallpackage': 'true',                               # Настройка автоматической установки отсутствующих библиотек
    'GUI'               : 'false',                              # Графический интерфейс пользователя
    'Font'              : 'Times 18',                           # Настройки шрифта
    'windowSize'        : '1000x600'                            # Размерность окна
}

cfg = StandartCfg

def CreateConfig(path = "settings.ini", conf = StandartCfg):
    config = configparser.ConfigParser()
    config.add_section("Settings")
    
    for sett in conf:
        config.set("Settings", sett, str(conf[sett]))
    
    with open(path, "w") as config_file:
        config.write(config_file)

def GetConfig(path = "settings.ini"):
    if not os.path.exists(path):
        CreateConfig(path)
        return 0

    config = configparser.ConfigParser()
    config.read(path)
    
    for sett in StandartCfg:
        if sett == "metrics":
            cfg[sett] = (config.get("Settings", sett)).split(',')
        else:
            value = config.get("Settings", sett)
            
            if value.isdigit():
                cfg[sett] = int(value)
            else: 
                cfg[sett] = value

print("Считываем файл конфигурации")
GetConfig()
print("Конфигурация считана")

if cfg['GUI'] == 'true':
    try:
        import tkinter
        tkin = True
        print('Tkinter подключен!')


    except ImportError:
        print('Модуль Tkinter не установлен, графический интерфейс и выбор файла модели недоступны\nПредпренимается попытка установить...')
        if cfg['autoInstallpackage']:
            try:
                mod_inst = subprocess.Popen("pip install tkinter", shell=True) 
                mod_inst.wait()
            except:
                print('Ошибка установки')
        tkin = False
    

try:
    from langdetect import detect
    ld = True
    print('langdetect подключен!')

except ImportError:
    print('Модуль langdetect не установлен, определение языка отключено\nПредпренимается попытка установить...')
    if cfg['autoInstallpackage']:
        try:
            mod_inst = subprocess.Popen("pip install langdetect", shell=True) 
            mod_inst.wait()
        except:
            print('Ошибка установки')
    ld = False

try: 
    import sklearn
    sl = True
    print('Sklearn подключен!')

except ImportError:
    print('Модуль sklearn не установлен!\nПредпренимается попытка установить...')
    if cfg['autoInstallpackage']:
        try:
            mod_inst = subprocess.Popen("pip install sklearn", shell=True) 
            mod_inst.wait()
        except:
            print('Ошибка установки')
    sl = False
try:
    import keras
    ks = True
    print('Keras подключен!')

except ImportError:
    print('Модуль Keras не установлен!\nПредпренимается попытка установить...')
    if cfg['autoInstallpackage']:
        try:
            mod_inst = subprocess.Popen("pip install --upgrade tensorflow", shell=True) 
            mod_inst.wait()
            mod_inst = subprocess.Popen("pip install keras", shell=True)
            mod_inst.wait()
        except:
            print('Ошибка установки')
    ks = False

try:
    import numpy as np
    nmp = True
    print('Numpy подключен!')

except ImportError:
    print('Модуль Numpy не установлен!\nПредпренимается попытка установить...')
    if cfg['autoInstallpackage']:
        try:
            mod_inst = subprocess.Popen("pip install numpy", shell=True) 
            mod_inst.wait()
        except:
            print('Ошибка установки')
    nmp = False

try: 
    import pandas as pd
    pand = True
    print('Pandas подключен!')

except ImportError:
    print('Модуль Pandas не установлен!\nПредпренимается попытка установить...')
    if cfg['autoInstallpackage']:
        try:
            mod_inst = subprocess.Popen("pip install pandas", shell=True) 
            mod_inst.wait()
        except:
            print('Ошибка установки')
    pand = False

try:
    import IzumrudLearner
    il = True
    print('Learner найден и подключен!')

except ImportError:
    print('Нет доступа к обучающему модулю программы')
    il = False

if (not pand or not ks or not nmp or not pand or not il):
    print('Работа программы не может быть продолжена из-за нехватки элементов. Установите элементы указанные выше!')
    if cfg['autoInstallpackage']:
        print('Автоустановка была включена, пакеты были установлены. Перезапустите программу')
    time.sleep(2)
    exit()


def modelLoad(place = os.getcwd() + cfg['modelPlace']):
    mod = keras.models.load_model(place)
    print("загружена модель")
    return mod

model = modelLoad()
tokenizer = keras.preprocessing.text.Tokenizer(num_words=cfg['NUM'])
token = False
rdyToLearn = False

def prepareTokenizer():
    global rdyToLearn
    global token
    try:
        if cfg['GUI']:
            infoLabel.config(text = 'Подготовка токенайзера')
    except NameError:
        pass
    finally:
        print("Подготовка токенайзера")

    n = ['id', 'date', 'name', 'text', 'typr', 'rep', 'rtw', 'faw', 'stcount', 'foll', 'frien', 'listcount']
    data_positive = pd.read_csv(os.getcwd() + cfg['positivePlace'], sep=';', error_bad_lines=False, names=n, usecols=['text'])
    data_negative = pd.read_csv(os.getcwd() + cfg['negativePlace'], sep=';', error_bad_lines=False, names=n, usecols=['text'])

    sample_size = min(data_positive.shape[0], data_negative.shape[0])

    raw_data = np.concatenate((data_positive['text'].values[:sample_size],
                            data_negative['text'].values[:sample_size]))

    data = [IzumrudLearner.preprocess_text(t) for t in raw_data]

    try:
        if cfg['GUI']:
            infoLabel.config(text = 'Фитим токенайзер')
    except NameError:
        pass
    finally:
        print("фитим токенайзер")

    tokenizer.fit_on_texts(data)
    token = True
    rdyToLearn = True

    try:
        if cfg['GUI']:
            infoLabel.config(text = 'Готово')
    except NameError:
        pass
    finally:
        print("Готово")

    time.sleep(1.5)
    try:
        if cfg['GUI']:
            infoLabel.config(text = '')
    except NameError:
        pass

def chooseOption(txt = 'Выберите опцию: '):
    try:
        escText = input(txt)
        if escText.isdigit():
            return int(escText)
        else:
            return -1
    except:
        return -1

def description():
    PrintBasics("Описание")
    print("Программное приложение анализа текста с использованием технологии Data Science.\nПредназначено для анализа текста длинной до 28 слов и обучения моделей способных с высокой точностью определять тональность текстов\nАвтор: Шитов Виктор\n")
    input("Нажмите любую кнопку чтобы вернуться в меню...")

def analysis():
    PrintBasics("Анализ Текста")
    
    text = input("Введите текст для анализа \nЧтобы выйти из меню анализа введите 0\n")
    
    if (text == "0"):
        return
    try:
        if (ld):
            lang = detect(text)
            if (lang == 'ru'):
                print("Язык - Русский")
            if (lang == 'en'):
                print("Язык - Английский")
        else:
            print('langdetect не установлен, язык не определен')
        text = IzumrudLearner.preprocess_text(text)
        numWord = len(text)
        print(f"Количество слов - {numWord}")
        text = IzumrudLearner.get_sequences(tokenizer,text, cfg['Sen_Lenght'])
        if lang == 'en': 
            print('Модель не обучена под английский язык, точность нивелируется.')
        vecsum = float(sum(model.predict(text)))
        vecsum = standSigmo(vecsum, numWord)
        print('Сумма векторов - ' + "%.2f" % vecsum)
        if vecsum >= float(cfg['vecsumMax']):
            print("Текст положительный")
        if vecsum <= float(cfg['vecsumMin']):
            print("Текст отрицательный")
        elif (vecsum < float(cfg['vecsumMax']) and vecsum > float(cfg['vecsumMin'])):
            print('Текст нейтральный')
        input("Нажмите любую кнопку чтобы вернуться в меню...")
    except:
        print("Введен текст который не может быть проанализирован либо ничего введено небыло, Повторите ввод...")
        print('Ошибка:\n', traceback.format_exc())
        input(' Нажмите любую кнопку чтобы продолжить...')
        analysis()

def modelLearner(erText = ''):
    PrintBasics('Обучение модели', erText)
    print("""
    1. Запустить обучение модели по готовым параметрам
    2. Запустить обучение модели по настраиваемым параментрам
    3. Отобразить стандартные параметры обучения
    4. Вернутся в меню
    """)

    inp = chooseOption()
    
    if (inp != -1):
        if (inp == 1):
            try:
                IzumrudLearner.learn_model_RNN(cfg, tokenizer)
                time.sleep(2)
                modelLearner('Модель обучена успешно')
            except:
                modelLearner(traceback.format_exc() + '\nВозникла ошибка при обучении модели, проверьте программный код!')

        elif (inp == 2):
            epoch = chooseOption('Введите количество эпох обучения или 0 чтобы прекратить: ')
            if (epoch <= 0):
                modelLearner('была прервана настройка')

            neurons = chooseOption('Введите количество нейронов или 0 чтобы прекратить: ')
            if (neurons <= 0):
                modelLearner('была прервана настройка')

            choosen = cfg
            choosen['epoch'] = epoch
            choosen['RNNneurons'] = neurons
            try:
                IzumrudLearner.learn_model_RNN(choosen, tokenizer)
                time.sleep(2)
                modelLearner('Модель обучена успешно')
            except:
                modelLearner('Возникла неизвестная ошибка при обучении')
        elif (inp == 3):
            modelLearner(f"""
    {cfg['epoch']}\t\t - Количество эпох обучения
    {cfg['Sen_Lenght']}\t\t - Количество слов в наборе (Лучше не менять)
    {cfg['RNNneurons']}\t\t - Количество нейронов в рекурентной сети
    {cfg['optimizer']}\t - Оптимизатор (Лучше не менять)
    {cfg['vecsumMax']}\t\t - Верхний предел суммы векторов выше которого текст считается положительным
    {cfg['vecsumMin']}\t\t - Нижний предел суммы векторов ниже которого текст считается отрицательным
            """)
        elif (inp == 4):
            return 0
        else:
            modelLearner('Выбрана опция отсутствующая в списке')
    else:
        modelLearner('Ошибка при вводе')

def PrintBasics(optionName = "", error = ""):
    os.system('cls' if os.name == 'nt' else 'clear')

    print(f"""
    \033[32m{cfg['name']}\033[0m Версия:{cfg['version']}

    \033[32m╔══╗╔═══╗╔╗──╔╗╔══╗\033[0m   ─╔╗──╔══╗──╔══╗\033[32m   //̅ _ ̅\\\\
    \033[32m╚╗╔╝╚═╗─║║║──║║║╔╗║\033[0m   ╔╝║──║╔╗║──║╔╗║\033[32m  || /   \\ ||
    \033[32m─║║──╔╝╔╝║╚╗╔╝║║╚╝║\033[0m   ╚╗║──║║║║──║║║║\033[32m  |||     |||
    \033[32m─║║─╔╝╔╝─║╔╗╔╗║║╔╗║\033[0m   ─║║──║║║║──║║║║\033[32m  |||     |||
    \033[32m╔╝╚╗║─╚═╗║║╚╝║║║║║║\033[0m   ─║║╔╗║╚╝║╔╗║╚╝║\033[32m  || \\   / ||
    \033[32m╚══╝╚═══╝╚╝──╚╝╚╝╚╝\033[0m   ─╚╝╚╝╚══╝╚╝╚══╝\033[32m   \\\\_ ̅ _// 
    \033[0m""")

    print(error)
    print("──────────────────────────────────────────────────────────")
    print(optionName)
    print("──────────────────────────────────────────────────────────")



def noGui():
    prepareTokenizer()
    global errorText
    errorText = ""
    global model

    while 0==0:
        PrintBasics("Опции", errorText)
        errorText = ""

        print("""
        1. Описание
        2. Анализировать текст
        3. Обучить новую модель
        4. Обновить конфигурацию
        5. Выход
        """)

        inp = chooseOption()

        if (inp != -1):
            if inp == 1:
                description()
                continue
            elif inp == 2:
                analysis()
                continue
            elif inp == 3:
                modelLearner()
                continue
            elif inp == 4:
                GetConfig()
                for sett in cfg:
                    print("{} {}".format(sett.ljust(30, '_'), cfg[sett]))
                model = modelLoad()
                print('Модель обновлена')
                input("Нажмите любую кнопку чтобы продолжить...")
            elif inp == 5:
                print('спасибо за внимание')
                time.sleep(1.5)
                break
            else:
                errorText = 'выбрана опция отсутствующая в списке.'
        else:
            errorText = 'ошибка при вводе, повторите ввод.'

def sqrtSigmo(x,y):
    return float(x/math.sqrt(y/10+math.pow(x,2)))

def standSigmo(x,y):
    return float(1/(1+math.exp((-x*30)/y + 5)))

def Ganalysis(textBlock, tonal, WordNum, lan, vec):
    global token
    text = textBlock.get()
    if token:
        try:
            if (ld):
                lang = detect(text)
                if (lang == 'ru'):
                    lan['text'] = "Язык - Русский"
                if (lang == 'en'):
                    lan['text'] = "Язык - Английский"
            else:
                lan['text'] = "Langdetect не установлен"
            
            text = IzumrudLearner.preprocess_text(text)
            numWord = len(text)
            WordNum['text'] = 'Количество слов - ' + str(numWord)
            text = IzumrudLearner.get_sequences(tokenizer,text, cfg['Sen_Lenght'])
            vecsum = float(sum(model.predict(text)))
            vecsum = standSigmo(vecsum,numWord)
            vec['text'] = 'Сумма векторов - ' + '%.2f' % vecsum

            if vecsum >= float(cfg['vecsumMax']):
                tonal['text'] = 'Текст - положительный'
            if vecsum <= float(cfg['vecsumMin']):
                tonal['text'] = 'Текст - отрицательный'
            elif (vecsum < float(cfg['vecsumMax']) and vecsum > float(cfg['vecsumMin'])):
                tonal['text'] = 'Текст - нейтральный'
            
            infoLabel['text'] = 'Успешно проанализирован текст'
        except:
            infoLabel['text'] = 'Возникла ошибка при анализе текста'
    else:
        infoLabel['text'] = 'Токенайзер еще не готов'

def Ganalys(menu):
    menu.destroy()
    menu = tkinter.Frame(width = 550, height = 480)

    label1      = tkinter.Label(menu, text = "Анализ текста", font = cfg['Font'])
    analysText  = tkinter.Entry(menu, width = 50, font = cfg['Font'])
    tonal       = tkinter.Label(menu, text = "Тональность текста", font = cfg['Font'])
    WordNum     = tkinter.Label(menu, text = 'Количество слов', font = cfg['Font'])
    lan         = tkinter.Label(menu, text = 'Язык текста', font = cfg['Font'])
    vec         = tkinter.Label(menu, text = 'Сумма векторов', font = cfg['Font'])
    backButt    = tkinter.Button(menu, text = 'Вернуться', command = lambda : mainMenu(menu), font = cfg['Font'])
    processButt = tkinter.Button(menu, text = 'Анализировать', width = 15, command = lambda : Ganalysis(analysText, tonal, WordNum, lan, vec), font = cfg['Font'])

    match = re.findall(r'(?=.)\d{1,}', cfg['modelPlace'])

    params      = tkinter.Label(menu, 
    text = "Параметры модели:\n {} {}\n {} {}\n {} {}\n {} {}".format("Ограничение количества слов", str(cfg['Sen_Lenght']), "Количество нейронов", match[1], 'Количество эпох обучения', match[0], 'Точность после обучения', match[2]), 
    relief = 'raise', 
    bd = 1, 
    justify = 'left',
    width = 43, 
    font = cfg['Font'])

    menu.pack(expand = 1, fill = 'both')

    label1.place(relx = 0.5, rely = 0.05)
    analysText.place(relx = 0.1, rely = 0.2)
    processButt.place(relx = 0.4168, rely = 0.28)
    params.place(relx = 0.1, rely = 0.4)
    tonal.place(relx = 0.75, rely = 0.2)
    WordNum.place(relx = 0.75, rely = 0.3)
    lan.place(relx = 0.75, rely = 0.4)
    vec.place(relx = 0.75, rely = 0.5)
    backButt.place(relx = 0.85, rely = 0.9)

def startGlearn(conf, epocholder = None, neuronsholder = None):
    global rdyToLearn
    if rdyToLearn:
        Learn = Thread(target = Glearn, args = (conf,epocholder,neuronsholder,), daemon = True)
        Learn.start()
    else:
        infoLabel['text'] = "Модуль еще не готов"

def Glearn(conf, epocholder = None, neuronsholder = None):
    global rdyToLearn
    rdyToLearn = False
    if not epocholder == None:
        try:
            value = int(epocholder.get())
            if value > 0 :
                conf['epoch'] = int(epocholder.get())
            else:
                infoLabel['text'] = 'Введено слишком малое значение!'
                rdyToLearn = True
                return 0
        except:
            infoLabel['text'] = 'Ошибка при считывании данных'
            rdyToLearn = True
            return 0
    if not epocholder == None:
        try:
            value = int(neuronsholder.get())
            if value > 0 :
                conf['RNNneurons'] = value
            else:
                infoLabel['text'] = 'Введено слишком малое значение!'
                rdyToLearn = True
                return 0
        except:
            infoLabel['text'] = 'Ошибка при считывании данных'
            rdyToLearn = True
            return 0
    try:
        IzumrudLearner.learn_model_RNN(conf, tokenizer, infoLabel)
    except:
        infoLabel['text'] = 'Возникла ошибка при обучении модели!'
        print(traceback.format_exc())
    rdyToLearn = True


def Glearner(menu):
    menu.destroy()

    GetConfig()

    menu        = tkinter.Frame(width = 550, height = 480)
    backButt    = tkinter.Button(menu, text = 'Вернуться', command = lambda : mainMenu(menu), font = cfg['Font'])

    Label           = tkinter.Label(menu, text = 'Обучение без параметров')
    Label2          = tkinter.Label(menu, text = 'Задать параметры')
    checktf         = tkinter.Checkbutton(menu, state = 'disabled', text = 'TensorFlow', font = cfg['Font'])
    checkks         = tkinter.Checkbutton(menu, state = 'disabled', text = 'Keras', font = cfg['Font'])
    checksk         = tkinter.Checkbutton(menu, state = 'disabled', text = 'Scikit-learn', font = cfg['Font'])
    checknp         = tkinter.Checkbutton(menu, state = 'disabled', text = 'Numpy', font = cfg['Font'])
    checkpd         = tkinter.Checkbutton(menu, state = 'disabled', text = 'Pandas', font = cfg['Font'])
    checkil         = tkinter.Checkbutton(menu, state = 'disabled', text = 'Izumrud Learner', font = cfg['Font'])
    noParamsLearn   = tkinter.Button(menu, text = 'Обучить', width = 10, height = 2, command = lambda : startGlearn(cfg), font = cfg['Font'])
    epochLabel      = tkinter.Label(menu, text = 'Количество эпох', font = cfg['Font'])
    epochEnter      = tkinter.Entry(menu)
    neuronsLabel    = tkinter.Label(menu, text = 'Количество нейронов', font = cfg['Font'])
    neuronsEnter    = tkinter.Entry(menu)
    paramsLearn     = tkinter.Button(menu, text = 'Обучить', width = 10, height = 2, command = lambda : startGlearn(cfg, epochEnter,neuronsEnter), font = cfg['Font']) 


    curParam        = tkinter.Label(menu, 
    text = f"""\tПараметры
Количество эпох: {cfg['epoch']}
Длина предложения: {cfg['Sen_Lenght']}
Количество нейронов: {cfg['RNNneurons']}
Оптимизитаор: {cfg['optimizer']}
Расчет ошибки: {cfg['lossType']}
Верхняя сумма векторов: {cfg['vecsumMax']}
Нижняя сумма векторов: {cfg['vecsumMin']}
    """, justify = "left", font = cfg['Font'])

    if ks:
        checktf.select()
        checkks.select()

    if sl:
        checksk.select()
    
    if nmp:
        checknp.select()
    
    if pand:
        checkpd.select()
    
    if il:
        checkil.select()


    Label2.place(relx = 0.65, rely = 0.05)
    Label.place(relx = 0.15, rely = 0.05)
    curParam.place(relx = 0.3, rely = 0.18)
    checktf.place(relx = 0.05, rely = 0.2)
    checkks.place(relx = 0.05, rely = 0.25)
    checksk.place(relx = 0.05, rely = 0.3)
    checknp.place(relx = 0.05, rely = 0.35)
    checkpd.place(relx = 0.05, rely = 0.4)
    checkil.place(relx = 0.05, rely = 0.45)
    noParamsLearn.place(relx = 0.15, rely = 0.55)
    epochLabel.place(relx = 0.7, rely = 0.25)
    epochEnter.place(relx = 0.7, rely = 0.3)
    neuronsLabel.place(relx = 0.7, rely = 0.4)
    neuronsEnter.place(relx = 0.7, rely = 0.45)
    paramsLearn.place(relx = 0.7, rely = 0.55)

    menu.pack(expand = 1, fill = 'both')

    
    backButt.place(relx = 0.85, rely = 0.9)

def mainMenu(menu):
    menu.destroy()

    menu = tkinter.Frame()

    deco        = tkinter.Label(menu, image = img)
    descript    = tkinter.Button(menu, text = "Описание", width = 40, command = lambda : Gdiscription(menu), font = cfg['Font'])
    analys      = tkinter.Button(menu, text = "Анализ текста", width = 40, command = lambda : Ganalys(menu), font = cfg['Font'])
    learn       = tkinter.Button(menu, text = "Обучение модели", width = 40, command = lambda : Glearner(menu), font = cfg['Font'])
    ehit        = tkinter.Button(menu, text = "Выход", width = 40, command = exit, font = cfg['Font'])

    menu.pack       (expand = 1)
    descript.grid   (row = 0, column = 0, ipady = 15)
    analys.grid     (row = 1, column = 0, ipady = 15)
    learn.grid      (row = 2, column = 0, ipady = 15)
    ehit.grid       (row = 3, column = 0, ipady = 15)
    deco.grid       (row = 0, column = 1, rowspan = 4, ipadx = 40)


def Gdiscription(menu):
    menu.destroy()
    menu = tkinter.Frame()

    text = tkinter.Label(menu, text = "Программное приложение анализа текста с использованием технологии Data Science.\nПредназначено для анализа текста длинной до 28 слов и обучения моделей\nспособных с высокой точностью определять тональность текстов\nАвтор: Шитов Виктор\n", justify = "left", font = cfg['Font'])
    back = tkinter.Button(menu, text = 'Вернуться', command = lambda : mainMenu(menu), font = cfg['Font'])
    menu.pack(expand = 1)
    text.pack()
    back.pack(anchor = 'se')


if cfg['GUI'] == 'true':
    root = tkinter.Tk()
    root.iconbitmap("resources/pictures/item.ico")
    root.title(cfg['name'])
    root.resizable(False, False)
    root.geometry(cfg['windowSize'])
    img = tkinter.PhotoImage(file = os.getcwd() + '/resources/pictures/main.png')
    
    menu = tkinter.Frame()

    infoLabel = tkinter.Label(text = "Информационная панель")
    infoLabel.pack(anchor = "se")

    mainMenu(menu)

    Proc = Thread(target = prepareTokenizer, daemon = True)
    Proc.start()

    root.mainloop()

else:
    noGui()
    

    
    


