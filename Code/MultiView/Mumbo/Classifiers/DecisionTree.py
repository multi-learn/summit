from sklearn import tree
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
# from sklearn.multiclass import OneVsRestClassifier
from ModifiedMulticlass import OneVsRestClassifier

# Add weights 

def DecisionTree(data, labels, arg, weights):
    depth = int(arg[0])
    isBad = False
    classifier = tree.DecisionTreeClassifier(max_depth=depth)
    #classifier = OneVsRestClassifier(tree.DecisionTreeClassifier(max_depth=depth))
    classifier.fit(data, labels, weights)
    prediction = classifier.predict(data)
    pTr, r, f1, s = precision_recall_fscore_support(labels, prediction, sample_weight=weights)
    if np.mean(pTr) < 0.5:
        isBad = True

    return classifier, prediction, isBad, pTr

def getConfig(classifierConfig):
    depth = classifierConfig[0]
    return 'with depth ' + depth + '.'

