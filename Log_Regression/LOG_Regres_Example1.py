"""
SOURCE = https://realpython.com/logistic-regression-python/
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix


x = np.arange(10).reshape(-1,1)
y = np.array([0,0,0,0,1,1,1,1,1,1])

#print(x)
#print(y)

model = LogisticRegression(solver='liblinear',random_state=0)

"""
Params:
    penalty = строка определяющая тип регурилизации 
    dual = True\False (Primal or Dual formulation)
    tol = tolerance for stopping proc
    c = strength of regularization
    fit_intercept = True\False (Calculate intecept or consider it equal Zero)
    intercept_scaling = Scale of intercept
    class_weight = None - All class have weight one ; Balanced - Related to each class
    random_state = pseudo random generator
    solver = what solve to use for fitting model

    max_iter = Means what it means

    ***I don't wanna to rewtire each param , check source if you need

"""

model.fit(x,y)
"""
print(model.classes_) 
print(model.intercept_) 
print(model.coef_) 

"""

#print(model.predict_proba(x))
"""
At this point i think , that this guide is kinda mid. Just an example with some formula's??
"""
#print(model.predict(x))
print(model.score(x,y))

"""
Further in guide they just using confusion matrix and class report to see how well our model do.
Buuut. I prefer to skip this. Let's check Example 2.
"""