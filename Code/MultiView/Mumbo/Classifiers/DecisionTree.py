from sklearn import tree
from sklearn.multiclass import OneVsRestClassifier

def decisionTree(data, labels, arg):
    classifier = OneVsRestClassifier(tree.DecisionTreeClassifier(max_depth=arg))
    classifier.fit(data, labels)
    return classifier, classifier.predict(data)