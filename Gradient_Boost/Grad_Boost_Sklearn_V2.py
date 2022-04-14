import pandas as pd
import numpy as np

# Более без инструменатальный способ Бустинга с использование Линейной

from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt


"""
GUIDE_URL = https://www.youtube.com/watch?v=ZNJ3lKyI-EY
"""

X, y  = make_regression(n_samples=10,n_features=3)

df = pd.DataFrame(X)
df['y_true'] = y

"""
Внутренности и пошаговое исполнение бустинга без использование цикла

df['y_pred_0'] = df['y_true'].mean() # Наше среднее предсказание на все
df['residual_0'] = df['y_true'] - df['y_pred_0'] # Насколько наше предсказание далеко от истинного


tree_1 = DecisionTreeRegressor(max_depth=1) # "Пенек"

tree_1.fit(df[[0,1,2]],df['residual_0'])

df['tree_pred_1'] = tree_1.predict(df[[0,1,2]])

#fig = plt.figure(figsize=(4,5))
#visual =  plot_tree(tree_1,filled=True) # Визуализация деревьев

# Процесс бустинга
nu = 0.1 #Learning rate
df['y_pred_1'] = df['y_pred_0'] + nu * df['tree_pred_1']

df['residual_1'] = df['y_true'] - df['y_pred_1']

tree_2 = DecisionTreeRegressor(max_depth=1)
tree_2.fit(df[[0,1,2]],df['residual_1'])

df['y_pred_2'] = df['y_pred_1'] + nu * tree_2.predict(df[[0,1,2]])

#plt.show()

print(mean_absolute_error(df['y_true'],df['y_pred_0'])) #Первое предсказание нашего дерева
print(mean_absolute_error(df['y_true'],df['y_pred_1'])) #Второе
print(mean_absolute_error(df['y_true'],df['y_pred_2'])) #Третье
"""
#Бустинг с циклом (не идеальный алгоритм)
n = 10 # Количество деревьев
nu = 0.1
trees = [] # Список с деревьям
df['y_pred'] = df['y_true'].mean() # Константное предсказание


for i in range(n):
    df['residual'] = df['y_true'] - df['y_pred']

    tree = DecisionTreeRegressor(max_depth=1)
    tree.fit(df[[0,1,2]],df['residual'])

    df['y_pred'] = df['y_pred'] + nu * tree.predict(df[[0,1,2]])
    # df['y_pred] += nu * tree.predict(df[[0,1,2]]) - Аналог строки выше

    trees.append(tree)
    print(mean_absolute_error(df['y_true'],df['y_pred']))

#print(df)


test = df[[0,1,2]].copy()
test['y_pred'] = df['y_true'].mean() 

for tree in trees:
    test['y_pred'] += nu * tree.predict(df[[0,1,2]])

print(test)

