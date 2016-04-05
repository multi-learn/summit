from sklearn import tree
from sklearn.multiclass import OneVsRestClassifier
# from ModifiedMulticlass import OneVsRestClassifier

# Add weights 

def DecisionTree(data, labels, arg, weights):
    classifier = OneVsRestClassifier(tree.DecisionTreeClassifier(max_depth=arg))
    classifier.fit(data, labels)
    return classifier, classifier.predict(data)

