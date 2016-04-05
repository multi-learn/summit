import Mumbo.Mumbo as Mumbo
import GetMutliviewDb as DB

NB_VIEW = 4
DATASET_LENGTH = 300
NB_CLASS = 4
NB_ITER = 50
classifierName="DecisionTree"
NB_CORES = 4
classifierConfig = 3

DATASET, VIEW_DIMENSIONS, CLASS_LABELS = DB.createFakeData(NB_VIEW, DATASET_LENGTH, NB_CLASS)
bestClassifiers, generalAlphas, bestViews = Mumbo.trainMumbo(DATASET, CLASS_LABELS, NB_CLASS, NB_VIEW, NB_ITER, DATASET_LENGTH, classifierName, NB_CORES, classifierConfig)

predictedLabels = Mumbo.classifyMumbo(DATASET, bestClassifiers, generalAlphas, bestViews, NB_CLASS)
