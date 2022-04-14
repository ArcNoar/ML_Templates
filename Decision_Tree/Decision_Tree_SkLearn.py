"""
GUIDE_URL(Theory) = https://www.youtube.com/watch?v=7VeUPuFGJHk
"""

import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.model_selection import train_test_split # Spliting data to training and testing sets
from sklearn.model_selection import cross_val_score # Cross Validation itself

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

# DATA PREPARATION

pd.set_option('display.max_columns', None) # Sets Unlimited columns to display

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",header=None)

df.columns = [
    'age',
    'sex',
    'cp',
    'restbp',
    'chol',
    'fbs',
    'restecg',
    'thalach',
    'exang',
    'oldpeak',
    'slope',
    'ca',
    'thal',
    'hd'
    ]


#print(df.dtypes) # вывод типов данных столбцов

#print(df.head())

#print(df['ca'].unique())
#print(df['thal'].unique()) "?" -  недостающие данные.


#print(df.loc[(df['ca'] == '?')
#            |
#           (df['thal'] == '?')]) # Считает количество строк с недостающими данными.


#print(df)

"""
В общем, 6 строк с плохими данным по сравнению с 303 общими строками, это мелочь, так что вместо подгонки и редактирования значений
этих строк, мы тупо удалим их.
*** не универсальный подход ***
"""

df_no_missing = df.loc[(df['ca'] != '?')
                       &
                       (df['thal'] != '?')]


#print(len(df_no_missing)) # = 297 (prev = 303)


# DATA FORMATING - p1
# Делаем копию столбцов исключая (hd)
X = df_no_missing.drop('hd', axis=1).copy() #Alt: X = df_no_missing.iloc[:,:-3]
#print(X.head())

# Че то делаем я устал. Создаем копию данных для предсказания.
y = df_no_missing['hd'].copy()
#print(y.head())

# Part2 - One-Hot Encoding
"""
Float - 0~1
Category : 1-catg,2-catg,3-catg

age - float
sex - category
cp - category
restbp - float
chol - float
fbs - category
thalach - float
exang - category
oldpeak - Float
slope - category
ca - Float
thal - Category

"""
"""
В общем, наше дерево решений не поддерживает напрямую категориальный данные, так что мы переведем категориальный тип в другой формат
Для того чтобы наше древо могло с этим работать.
В видео было уточнено что есть варик использовать в формате непрерывных значений, но это может вызвать конфликт схожести между ними,
а це нам не особо нужно, и наш метод позволяет сделать каждую категорию независимой от другой, и их сходимость будет равной.
(Хотя если честно я не уверен что это подход действительно хорош.)
"""

#print(X['cp'].unique())

"""
В видео рассматривались пару способов экоддинга.
от SkLearn - ColumnTransformer()
и от Pandas - get_dummies()

Первый метод более практичен, потому что создает новый список, и категоричен под заданные данные. П.П - Категория цветов (RGB)
В случае ввода значения Orange, выбросит ошибку ибо такого типа данных не предусметрено в качестве выбора (Опционально).
Но такой метод не сохраняет названия столбцов которые мы задали. (Что кстати не такая большая проблема в реальной ситуации.)
"""

#print(pd.get_dummies(X, columns=['cp']).head())
"""
Мы работаем с отформатированным качественными данными, в реальной же ситуации стоит убедиться в том что каждый из столбцов
содержит только принятые категории.
"""

X_encoded = pd.get_dummies(X,columns=['cp',
                                      'restecg',
                                      'slope',
                                      'thal',
                                      ])


#print(x_encoded.head())

"""
Теперь пройдемся по двоичным данным вроде пола. 0~1
*** Опять же, в реально ситуации нужно убедиться что в данных присутствуют только 0 и 1
"""

#print(y.unique())
"""
Мы приводим все данные к 0 и 1, по причине того что данный пример это примитивное дерево классификации есть ли пациента проблемы с сердцем
или нет.
Так что мы можем себе позволить такое привидение данных.
"""

y_not_zero_index = y > 0 # get the index from each nonzero value in y
y[y_not_zero_index] = 1 # set each nonzero value in y to 1 

"""
PRELIM CLASS TREE
"""

X_train, X_test, y_train, y_test = train_test_split(X_encoded,y,random_state=42)

# Creating a Dec tree and fiting it

clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt = clf_dt.fit(X_train, y_train)

"""
# Визуализируем дерево.
plt.figure(figsize=(30,14))
plot_tree(clf_dt,
          filled=True,
          rounded=True,
          class_names=["No HD","Yes HD"],
          feature_names=X_encoded.columns)




# Confusion Matrix
# Такой метод вызова матрицы устарел, и стоит использовать  :
#  ConfusionMatrixDisplay.from_predictions 
#  ConfusionMatrixDisplay.from_estimator.

plot_confusion_matrix(clf_dt,X_test,y_test,display_labels=['Does not have HD','Has HD'])
#plt.show()
"""


"""
В общем оно так себе справилось, так что мы будем делать срез дерева.
"""

# COST COMPLEXITY PRUNING
path = clf_dt.cost_complexity_pruning_path(X_train, y_train) # Determine values for alpha
ccp_alphas = path.ccp_alphas # Extract different for alpha
ccp_alphas = ccp_alphas[:-1] # Exclude the maximum alpha

clf_dts = [] # Container for decision trees.

# now create one decision tree per value for alpha and store it in the array
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0,ccp_alpha=ccp_alpha)
    clf_dt.fit(X_train,y_train)
    clf_dts.append(clf_dt)
"""
# Теперь изобразим графиком эффективность деревьев
train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test,y_test) for clf_dt in clf_dts]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas,train_scores,marker='o',label="train",drawstyle="steps-post")
ax.plot(ccp_alphas,test_scores,marker='o',label="test",drawstyle="steps-post")
ax.legend()
plt.show()
"""
"""
График показал что наилучший результат происходит при альфе 0,016 (GINI)
"""


# CROSS VALIDATION

clf_dt = DecisionTreeClassifier(random_state=42,ccp_alpha=0.016) # Tree with alpha= 0.016
# Используем пяти секционную кросс валидацию чтобы потом протестить эффективность нашего подхода.

scores = cross_val_score(clf_dt,X_train,y_train,cv=5)
df = pd.DataFrame(data={'tree': range(5),'accuracy': scores})

#df.plot(x='tree',y='accuracy',marker='o',linestyle='--')

"""
Кросс валидация показала что наше значения альфы чувствительно к изменению данных, так что сейчас будем искать наилучшую альфу.
"""

alpha_loop_values = []
"""
В общем, сейчас мы пробежимся по всем альфа кандидатам кросс валидацией, потом сохраним их стандартное отклонение и все как бэ.
"""
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0,ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf_dt,X_train,y_train,cv=5)
    alpha_loop_values.append([ccp_alpha,np.mean(scores),np.std(scores)])



## СНОВА ВИЗУАЛИЗАЦИЯ 
alpha_results = pd.DataFrame(alpha_loop_values,
                             columns=['alpha','mean_accuracy','std'])
"""
alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   yerr='std',
                   marker='o',
                   linestyle='--')
"""

#plt.show()
"""
Крч наш аналил показал что наилучший альфа в диапозоне от 0.014 до 0.015
"""
ideal_ccp_alpha = alpha_results[(alpha_results['alpha'] > 0.014)
              &
              (alpha_results['alpha'] < 0.015)]['alpha'] # =>alpha= 0.014225~

# Конвертируем наше значение в приемлимый Float64
"""
*** Так же альфу можно попробовать найти поиском по сетке (но это уже совсем другая история)
*** Порядок действий в этом примере некорректный, кросс валидацию можно сделать и в самом начале.
"""

ideal_ccp_alpha = float(ideal_ccp_alpha)

# BUILDING
clf_dt_pruned = DecisionTreeClassifier(random_state=42,
                                       ccp_alpha=ideal_ccp_alpha)

clf_dt_pruned = clf_dt_pruned.fit(X_train,y_train)

plot_confusion_matrix(clf_dt_pruned,
                      X_test,
                      y_test,
                      display_labels=["Does not have HD","Has HD"])


plt.figure(figsize=(15,7.5))
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=['No HD','Yes HD'],
          feature_names=X_encoded.columns)


plt.show()




