from sklearn import tree
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
# from sklearn.multiclass import OneVsRestClassifier
from ModifiedMulticlass import OneVsRestClassifier
import random

# Add weights 

def getLabelSupports(CLASS_LABELS):
    labels = set(CLASS_LABELS)
    supports = [CLASS_LABELS.tolist().count(label) for label in labels]
    return supports, dict((label, index) for label, index in zip(labels, range(len(labels))))


def isUseful(nbTrainingExamples, index, CLASS_LABELS, labelDict):
    if nbTrainingExamples[labelDict[CLASS_LABELS[index]]] != 0:
        nbTrainingExamples[labelDict[CLASS_LABELS[index]]] -= 1
        return True, nbTrainingExamples
    else:
        return False, nbTrainingExamples


def subSample(data, labels, weights, subSampling):
    nbExamples = len(labels)
    labelSupports, labelDict = getLabelSupports(labels)
    nbTrainingExamples = [int(support * subSampling) for support in labelSupports]
    trainingExamplesIndices = []
    while nbTrainingExamples != [0 for i in range(len(labelSupports))]:
        index = int(random.randint(0, nbExamples - 1))
        isUseFull, nbTrainingExamples = isUseful(nbTrainingExamples, index, labels, labelDict)
        if isUseFull:
            trainingExamplesIndices.append(index)
    subSampledData = []
    subSampledLabels = []
    subSampledWeights = []
    for index in trainingExamplesIndices:
        subSampledData.append(data[index])
        subSampledLabels.append(labels[index])
        subSampledWeights.append(weights[index])
    return np.array(subSampledData), np.array(subSampledLabels), np.array(subSampledWeights)

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
    labelsSet = set(prediction)
    pTr, r, f1, s = precision_recall_fscore_support(labels, prediction, sample_weight=weights)
    if np.mean(pTr) < 0.5:
        isBad = True

    return classifier, prediction, isBad, pTr

def getConfig(classifierConfig):
    depth = classifierConfig[0]
    subSampling = classifierConfig[1]
    return 'with depth ' + depth + ' -' + ' sub-sampled at ' + subSampling + ' '

