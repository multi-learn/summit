import numpy as np


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


def subSample(data, labels, subSampling, randomState, weights=None):
    if weights is None:
        weights = np.ones(len(labels)) / len(labels)
    nbExamples = len(labels)
    labelSupports, labelDict = getLabelSupports(labels)

    nbTrainingExamples = [int(support * subSampling) if int(support * subSampling) > 0 else 1
                          for support in labelSupports]
    trainingExamplesIndices = []
    usedIndices = []
    while nbTrainingExamples != [0 for i in range(len(labelSupports))]:
        index = int(randomState.randint(0, nbExamples - 1))
        isUseFull, nbTrainingExamples = isUseful(nbTrainingExamples, index, labels, labelDict)
        if isUseFull and index not in usedIndices:
            trainingExamplesIndices.append(index)
            usedIndices.append(index)
    subSampledData = []
    subSampledLabels = []
    subSampledWeights = []
    for index in trainingExamplesIndices:
        subSampledData.append(data[index])
        subSampledLabels.append(labels[index])
        subSampledWeights.append(weights[index])
    return np.array(subSampledData), np.array(subSampledLabels), np.array(subSampledWeights)
