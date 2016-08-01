from sklearn import tree
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
# from sklearn.multiclass import OneVsRestClassifier
from ModifiedMulticlass import OneVsRestClassifier

# Add weights 

def getLabelSupports(CLASS_LABELS):
    labels = set(CLASS_LABELS)
    supports = [CLASS_LABELS.tolist().count(label) for label in labels]
    return supports, dict((label, index) for label, index in zip(labels, range(len(labels))))


def isUseful(labelSupports, index, CLASS_LABELS, labelDict):
    if labelSupports[labelDict[CLASS_LABELS[index]]] != 0:
        labelSupports[labelDict[CLASS_LABELS[index]]] -= 1
        return True, labelSupports
    else:
        return False, labelSupports


def subSample(data, labels, weights, subSampling):
    nbExamples = len(data)
    labelSupports, labelDict = getLabelSupports(labels)
    nbTrainingExamples = [int(support * subSampling) for support in labelSupports]
    trainingExamplesIndices = []
    while nbTrainingExamples != [0 for i in range(NB_CLASS)]:
        index = np.random.randint(0, nbExamples - 1, sum(nbTrainingExamples))
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
        subSampledData, subSampledLabels, subSampledWeights = subSample (data, labels, weights, subSampling)
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
    return 'with depth ' + depth + '.'

