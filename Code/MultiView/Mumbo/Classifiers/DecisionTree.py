from sklearn import tree
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
# from sklearn.multiclass import OneVsRestClassifier
from ModifiedMulticlass import OneVsRestClassifier

# Add weights 

def DecisionTree(data, labels, arg, weights, CLASS_LABELS):
    isBad = False
    classifier = OneVsRestClassifier(tree.DecisionTreeClassifier(max_depth=arg))
    classifier.fit(data, labels, weights/np.sum(weights))
    prediction = classifier.predict(data)
    pTr, r, f1, s = precision_recall_fscore_support(CLASS_LABELS, prediction, sample_weight=weights)
    if np.mean(pTr)<0.5:
        isBad = True
    return classifier, prediction, isBad

