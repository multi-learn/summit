from sklearn import tree
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from ModifiedMulticlass import OneVsRestClassifier
from SubSampling import subSample
# Add weights 

def DecisionTree(data, labels, arg, weights):
    depth = int(arg[0])
    subSampling = float(arg[1])
    if subSampling != 1.0:
        subSampledData, subSampledLabels, subSampledWeights = subSample(data, labels, weights, subSampling)
    else:
        subSampledData, subSampledLabels, subSampledWeights = data, labels, weights
    isBad = False
    classifier = tree.DecisionTreeClassifier(max_depth=depth)
    #classifier = OneVsRestClassifier(tree.DecisionTreeClassifier(max_depth=depth))
    classifier.fit(subSampledData, subSampledLabels, subSampledWeights)
    prediction = classifier.predict(data)
    pTr, r, f1, s = precision_recall_fscore_support(labels, prediction, sample_weight=weights)
    if np.mean(pTr) < 0.5:
        isBad = True

    return classifier, prediction, isBad, pTr

def getConfig(classifierConfig):
    depth = classifierConfig[0]
    subSampling = classifierConfig[1]
    return 'with depth ' + depth + ', ' + ' sub-sampled at ' + subSampling + ' '

