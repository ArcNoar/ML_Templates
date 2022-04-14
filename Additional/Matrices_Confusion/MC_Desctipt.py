"""
URL_GUIDE = https://www.youtube.com/watch?v=Kdsp6soqA7o


Еще один способ оценик предсказания.
Проверяет правильно ли идентифицированы предсказания.

True Pos \ False Pos

True Neg \ False Neg

Example 1
"""
from sklearn import metrics

pred = ["T", "F", "T", "T", "F"] #predicted set of values

actual = ["F", "F", "F", "T", "T"] #actual set of values
CM = metrics.confusion_matrix(pred, actual, labels=["T", "F"]) #confusion matrix

print(CM)
report = metrics.classification_report(pred, actual, labels=["T", "F"]) #precision, recall, f1-score,etc
print(report)


