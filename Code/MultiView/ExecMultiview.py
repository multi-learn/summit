import Mumbo.Mumbo as Mumbo
import GetMutliviewDb as DB
import operator


NB_VIEW = 2
DATASET_LENGTH = 300
NB_CLASS = 3
NB_ITER = 10
classifierName="DecisionTree"
NB_CORES = 2
classifierConfig = 5
pathToAwa = "/home/doob/"
views = ['phog-hist', 'cq-hist']



DATASET, CLASS_LABELS, viewDictionnary, labelDictionnary = DB.getAwaData(pathToAwa, NB_CLASS, views)
DATASET_LENGTH = len(CLASS_LABELS)

# DATASET, VIEW_DIMENSIONS, CLASS_LABELS = DB.createFakeData(NB_VIEW, DATASET_LENGTH, NB_CLASS)
bestClassifiers, generalAlphas, bestViews = Mumbo.trainMumbo(DATASET, CLASS_LABELS, NB_CLASS, NB_VIEW, NB_ITER, DATASET_LENGTH, classifierName, NB_CORES, classifierConfig)

predictedLabels = Mumbo.classifyMumbo(DATASET, bestClassifiers, generalAlphas, bestViews, NB_CLASS)

def error(computedLabels, testLabels):
    error = sum(map(operator.ne, computedLabels, testLabels))
    return float(error) * 100 / len(computedLabels)

print error(predictedLabels, CLASS_LABELS)