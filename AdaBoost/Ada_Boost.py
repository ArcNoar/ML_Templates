"""
url_guide = https://www.youtube.com/watch?v=LsK-xG1cLYA
"""
"""
Example_Source = https://www.datacamp.com/community/tutorials/adaboost-classifier-python
"""
#Base
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split

#Additional
from sklearn.svm import SVC # Support Vector Classifier


iris = datasets.load_iris()

X = iris.data

y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

"""
Base_Estimator(BE) = Weak Learner to train model.
n_estimators = amount of weak learners
"""
"""
#Type 1 (Without BE and SVM)

abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
"""
#Type 2 (With BE and SVM)

svc = SVC(probability=True,kernel='linear')

abc = AdaBoostClassifier(n_estimators=50,base_estimator=svc,learning_rate=1)

model = abc.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test,y_pred))