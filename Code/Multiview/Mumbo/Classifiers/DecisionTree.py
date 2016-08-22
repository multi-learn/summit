from sklearn import tree
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from ModifiedMulticlass import OneVsRestClassifier
from SubSampling import subSample
# Add weights 

def DecisionTree(data, labels, arg, weights):
    depth = int(arg[0])
    subSampling = float(arg[1])
    if subSampling != 1.0:
        subSampledData, subSampledLabels, subSampledWeights = subSample(data, labels, subSampling, weights=weights)
    else:
        subSampledData, subSampledLabels, subSampledWeights = data, labels, weights
    isBad = False
    classifier = tree.DecisionTreeClassifier(max_depth=depth)
    #classifier = OneVsRestClassifier(tree.DecisionTreeClassifier(max_depth=depth))
    classifier.fit(subSampledData, subSampledLabels, subSampledWeights)
    prediction = classifier.predict(data)
    accuracy = accuracy_score(labels, prediction)
    if accuracy < 0.5:
        isBad = True

    return classifier, prediction, isBad, accuracy

def getConfig(classifierConfig):
    depth = classifierConfig[0]
    subSampling = classifierConfig[1]
    return 'with depth ' + str(depth) + ', ' + ' sub-sampled at ' + str(subSampling) + ' '

def gridSearch(data, labels):
    minSubSampling = 1.0/(len(labels)/2)
    bestSettings = []
    bestResults = []
    classifier = tree.DecisionTreeClassifier(max_depth=1)
    preliminary_accuracies = np.zeros(50)
    for i in range(50):
        subSampledData, subSampledLabels, subSampledWeights = subSample(data, labels, 0.05)
        classifier.fit(subSampledData, subSampledLabels)
        prediction = classifier.predict(data)
        preliminary_accuracies[i] = accuracy_score(labels, prediction)
    preliminary_accuracy = np.mean(preliminary_accuracies)
    if preliminary_accuracy < 0.50:
        for max_depth in np.arange(10)+1:
            for subSampling in sorted(np.arange(20, dtype=float)+1/20, reverse=True):
                if subSampling > minSubSampling:
                    accuracies = np.zeros(50)
                    for i in range(50):
                        if subSampling != 1.0:
                            subSampledData, subSampledLabels, subSampledWeights = subSample(data, labels, subSampling)
                        else:
                            subSampledData, subSampledLabels, = data, labels
                        classifier = tree.DecisionTreeClassifier(max_depth=max_depth)
                        classifier.fit(subSampledData, subSampledLabels)
                        prediction = classifier.predict(data)
                        accuracies[i] = accuracy_score(labels, prediction)
                    accuracy = np.mean(accuracies)
                    if 0.5 < accuracy < 0.60:
                        bestSettings.append([max_depth, subSampling])
                        bestResults.append(accuracy)
    else:
        preliminary_accuracies = np.zeros(50)
        if minSubSampling < 0.01:
            for i in range(50):
                subSampledData, subSampledLabels, subSampledWeights = subSample(data, labels, 0.01)
                classifier.fit(subSampledData, subSampledLabels)
                prediction = classifier.predict(data)
                preliminary_accuracies[i] = accuracy_score(labels, prediction)
        preliminary_accuracy = np.mean(preliminary_accuracies)
        if preliminary_accuracy < 0.50:
            for subSampling in sorted((np.arange(19, dtype=float)+1)/200, reverse=True):
                if minSubSampling < subSampling:
                    accuracies = np.zeros(50)
                    for i in range(50):
                        subSampledData, subSampledLabels, subSampledWeights = subSample(data, labels, subSampling)
                        classifier = tree.DecisionTreeClassifier(max_depth=1)
                        classifier.fit(subSampledData, subSampledLabels)
                        prediction = classifier.predict(data)
                        accuracies[i] = accuracy_score(labels, prediction)
                    accuracy = np.mean(accuracies)
                    if 0.5 < accuracy < 0.60:
                        bestSettings.append([1, subSampling])
                        bestResults.append(accuracy)
        else:
            for subSampling in sorted((np.arange(19, dtype=float)+1)/2000, reverse=True):
                accuracies = np.zeros(50)
                for i in range(50):
                    subSampledData, subSampledLabels, subSampledWeights = subSample(data, labels, subSampling)
                    if minSubSampling < subSampling:
                        classifier1 = tree.DecisionTreeClassifier(max_depth=1)
                        classifier1.fit(subSampledData, subSampledLabels)
                        prediction = classifier1.predict(data)
                        accuracies[i] = accuracy_score(labels, prediction)
                accuracy = np.mean(accuracies)
                if 0.5 < accuracy < 0.60:
                    bestSettings.append([1, subSampling])
                    bestResults.append(accuracy)

    assert bestResults!=[], "No good settings found for Decision Tree!"

    return getBestSetting(bestSettings, bestResults)


def getBestSetting(bestSettings, bestResults):
    diffTo52 = 100.0
    bestSettingsIndex = 0
    for resultIndex, result in enumerate(bestResults):
        if abs(52.5-result)<diffTo52:
            bestSettingsIndex = resultIndex

    return map(lambda p: round(p, 4), bestSettings[bestSettingsIndex])
#    return map(round(,4), bestSettings[bestSettingsIndex])