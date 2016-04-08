import Mumbo.Mumbo as Mumbo
import GetMutliviewDb as DB
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support


NB_CLASS = 5
NB_ITER = 40
classifierName="DecisionTree"
NB_CORES = 3
classifierConfig = 4
pathToAwa = "/home/doob/"
views = ['phog-hist', 'decaf', 'cq-hist']
NB_VIEW = len(views)
LEARNING_RATE = 1.0

print "Getting db ..."
# DATASET, CLASS_LABELS, viewDictionnary, labelDictionnary = DB.getAwaData(pathToAwa, NB_CLASS, views)
# target_names = [labelDictionnary[label] for label in labelDictionnary]
DATASET, CLASS_LABELS = DB.getDbfromCSV('/home/doob/OriginalData/')
print DATASET
DATASET_LENGTH = 40
NB_CLASS = 2
NB_VIEW = 3
CLASS_LABELS = np.array([int(label) for label in CLASS_LABELS])
# print target_names
# print labelDictionnary
DATASET_LENGTH = len(CLASS_LABELS)
trainData, trainLabels, testData,testLabels = DB.extractRandomTrainingSet(DATASET, CLASS_LABELS, LEARNING_RATE, DATASET_LENGTH, NB_VIEW)
DATASET_LENGTH = len(trainLabels)
# print len(trainData), trainData[0].shape, len(trainLabels)
print "Done."

print 'Training Mumbo ...'
# DATASET, VIEW_DIMENSIONS, CLASS_LABELS = DB.createFakeData(NB_VIEW, DATASET_LENGTH, NB_CLASS)
bestClassifiers, generalAlphas, bestViews = Mumbo.trainMumbo(trainData, trainLabels, NB_CLASS, NB_VIEW, NB_ITER, DATASET_LENGTH, classifierName, NB_CORES, classifierConfig)
print "Trained."

print "Predicting ..."
predictedTrainLabels = Mumbo.classifyMumbo(trainData, bestClassifiers, generalAlphas, bestViews, NB_CLASS)
predictedTestLabels = Mumbo.classifyMumbo(testData, bestClassifiers, generalAlphas, bestViews, NB_CLASS)
print 'Done.'
print 'Reporting ...'
predictedTrainLabelsByIter = Mumbo.classifyMumbobyIter(trainData, bestClassifiers, generalAlphas, bestViews, NB_CLASS)
predictedTestLabelsByIter = Mumbo.classifyMumbobyIter(testData, bestClassifiers, generalAlphas, bestViews, NB_CLASS)
print str(NB_VIEW)+" views, "+str(NB_CLASS)+" classes, "+str(classifierConfig)+" depth trees"
print "Best views = "+str(bestViews)
print "Is equal : "+str((predictedTrainLabels==predictedTrainLabelsByIter[NB_ITER-1]).all())

target_names=['moins', 'plus']
print "On train : "
print classification_report(trainLabels, predictedTrainLabels, target_names=target_names)
print "On test : "
print classification_report(testLabels, predictedTestLabels, target_names=target_names)


def error(testLabels, computedLabels):
    error = sum(map(operator.ne, computedLabels, testLabels))
    return float(error) * 100 / len(computedLabels)


x = range(NB_ITER)
trainErrors = []
testErrors = []
for iterTrain, iterTest in zip(predictedTrainLabelsByIter, predictedTestLabelsByIter):
    pTr, r, f1, s = precision_recall_fscore_support(trainLabels, iterTrain)
    pTe, r, f1, s = precision_recall_fscore_support(testLabels, iterTest)

    trainErrors.append(np.mean(pTr))
    testErrors.append(np.mean(pTe))


figure = plt.figure()
ax1 = figure.add_subplot(111)

ax1.set_title("Accuracy depending on iteration")


ax1.set_xlabel("Iteration Indice")
ax1.set_ylabel("Accuracy")
ax1.plot(x,trainErrors, c='red', label='Train')
ax1.plot(x,testErrors, c='black', label='Test')
leg = ax1.legend()

plt.show()
print 'Done.'