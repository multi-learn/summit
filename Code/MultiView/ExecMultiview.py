import Mumbo.Mumbo as Mumbo
import GetMutliviewDb as DB
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score


NB_VIEW = 3
DATASET_LENGTH = 300
NB_CLASS = 5
NB_ITER = 10
classifierName="DecisionTree"
NB_CORES = 3
classifierConfig = 3
pathToAwa = "/home/doob/"
views = ['phog-hist', 'cq-hist', 'decaf']
NB_VIEW = len(views)
LEARNING_RATE = 0.5


DATASET, CLASS_LABELS, viewDictionnary, labelDictionnary = DB.getAwaData(pathToAwa, NB_CLASS, views)
target_names = [labelDictionnary[label] for label in labelDictionnary]
# print target_names
# print labelDictionnary
DATASET_LENGTH = len(CLASS_LABELS)
trainData, trainLabels, testData,testLabels = DB.extractRandomTrainingSet(DATASET, CLASS_LABELS, LEARNING_RATE, DATASET_LENGTH, NB_VIEW)
DATASET_LENGTH = len(trainLabels)
# print len(trainData), trainData[0].shape, len(trainLabels)

# DATASET, VIEW_DIMENSIONS, CLASS_LABELS = DB.createFakeData(NB_VIEW, DATASET_LENGTH, NB_CLASS)
bestClassifiers, generalAlphas, bestViews = Mumbo.trainMumbo(trainData, trainLabels, NB_CLASS, NB_VIEW, NB_ITER, DATASET_LENGTH, classifierName, NB_CORES, classifierConfig)

predictedTrainLabels = Mumbo.classifyMumbo(trainData, bestClassifiers, generalAlphas, bestViews, NB_CLASS)
predictedTestLabels = Mumbo.classifyMumbo(testData, bestClassifiers, generalAlphas, bestViews, NB_CLASS)

predictedTrainLabelsByIter = Mumbo.classifyMumbobyIter(trainData, bestClassifiers, generalAlphas, bestViews, NB_CLASS)
predictedTestLabelsByIter = Mumbo.classifyMumbobyIter(testData, bestClassifiers, generalAlphas, bestViews, NB_CLASS)

print "On train : "
print classification_report(trainLabels, predictedTrainLabels, target_names=target_names)
print "On test : "
print classification_report(testLabels, predictedTestLabels, target_names=target_names)


def error(computedLabels, testLabels):
    error = sum(map(operator.ne, computedLabels, testLabels))
    return float(error) * 100 / len(computedLabels)


x = range(NB_ITER)
trainErrors = []
testErrors = []
for iterTrain, iterTest in zip(np.transpose(predictedTrainLabelsByIter), np.transpose(predictedTestLabelsByIter)):
    trainErrors.append(error(trainLabels, iterTrain))
    testErrors.append(error(testLabels, iterTest))


figure = plt.figure()
ax1 = figure.add_subplot(111)

ax1.set_title("Error depending on iteration")


ax1.set_xlabel("iteration Indice")
ax1.set_ylabel("Error")
ax1.plot(x,trainErrors, c='red', label='Train error')
ax1.plot(x,testErrors, c='black', label='Test Error')
leg = ax1.legend()

plt.show()
