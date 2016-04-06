from sklearn import tree
import numpy as np
# from sklearn.multiclass import OneVsRestClassifier
from ModifiedMulticlass import OneVsRestClassifier

# Add weights 

def DecisionTree(data, labels, arg, weights):
    classifier = OneVsRestClassifier(tree.DecisionTreeClassifier(max_depth=arg))
    classifier.fit(data, labels, weights/np.sum(weights))
    # print classifier.predict(data)
    return classifier, classifier.predict(data)

