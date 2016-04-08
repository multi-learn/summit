import numpy as np
from string import digits
import os


def getOneViewFromDB(viewName, pathToDB, DBName):
    view = np.genfromtxt(pathToDB + DBName +"-" + viewName, delimiter=';')
    return view


def getClassLabels(pathToDB, DBName):
    labels = np.genfromtxt(pathToDB + DBName + "-" + "ClassLabels.csv", delimiter=';')
    return labels


def getDataset(pathToDB, viewNames, DBName):
    dataset = []
    for viewName in viewNames:
        dataset.append(getOneViewFromDB(viewName, pathToDB, DBName))
    return np.array(dataset)


def createFakeData(NB_VIEW, DATASET_LENGTH, NB_CLASS):
    VIEW_DIMENSIONS = np.random.random_integers(5, 20, NB_VIEW)

    DATA = [
                        np.array([
                                     np.random.normal(0.0, 2, viewDimension)
                                     for i in np.arange(DATASET_LENGTH)])
                        for viewDimension in VIEW_DIMENSIONS]

    CLASS_LABELS = np.random.random_integers(0, NB_CLASS-1, DATASET_LENGTH)
    return DATA, VIEW_DIMENSIONS, CLASS_LABELS


def getAwaLabels(nbLabels, pathToAwa):
    file = open(pathToAwa + 'Animals_with_Attributes/classes.txt', 'U')
    linesFile = [''.join(line.strip().split()).translate(None, digits) for line in file.readlines()]
    awaLabels = [linesFile[label] for label in np.arange(nbLabels)]
    return awaLabels


def getAwaData(pathToAwa, nbLabels, views):
    awaLabels = getAwaLabels(nbLabels, pathToAwa)
    nbView = len(views)
    labelDictionnary = {i: awaLabels[i] for i in np.arange(nbLabels)}
    viewDictionnary = {i: views[i] for i in np.arange(nbView)}
    rawData = []
    labels = []
    nbExample = 0
    # ij = []
    for view in np.arange(nbView):
        viewData = []
        for label in np.arange(nbLabels):
            pathToExamples = pathToAwa + 'Animals_with_Attributes/Features/' + viewDictionnary[view] + '/' + \
                             labelDictionnary[label] + '/'
            examples = os.listdir(pathToExamples)
            if view == 0:
                nbExample += len(examples)
            for example in examples:
                exampleFile = open(pathToExamples + example)
                viewData.append([[float(coordinate) for coordinate in raw.split()] for raw in exampleFile][0])
                if view == 0:
                    labels.append(label)
        rawData.append(np.array(viewData))
    data = rawData
    # data = np.empty((nbExample, nbView), dtype=list)
    # for viewIdice in np.arange(nbView):
    #     for exampleIndice in np.arange(nbExample):
    #         data[exampleIndice, viewIdice] = rawData[viewIdice][exampleIndice]
    #         # data[exampleIndice, viewIdice] = {i:rawData[viewIdice][exampleIndice][i] for i in np.arange(len(rawData[viewIdice][exampleIndice]))}

    return data, labels, viewDictionnary, labelDictionnary

def splitData(data, labels, ratioForTrain):
    trainData = []
    testData = []
    trainLabels = []
    testLabels = []
    for viewIndice in range(len(data)):
        a = 1
    return

def extractRandomTrainingSet(DATA, CLASS_LABELS, LEARNING_RATE, DATASET_LENGTH, NB_VIEW):
    nbTrainingExamples = int(DATASET_LENGTH * LEARNING_RATE)
    trainingExamplesIndices = np.random.random_integers(0, DATASET_LENGTH, nbTrainingExamples)
    trainData, trainLabels = [], []
    testData, testLabels = [], []
    for viewIndice in range(NB_VIEW):
        trainD, testD = [], []
        trainL, testL = [], []
        for i in np.arange(DATASET_LENGTH):
            if i in trainingExamplesIndices:
                trainD.append(DATA[viewIndice][i])
                trainL.append(CLASS_LABELS[i])
            else:
                testD.append(DATA[viewIndice][i])
                testL.append(CLASS_LABELS[i])
        trainData.append(np.array(trainD))
        testData.append(np.array(testD))
    trainLabels.append(np.array(trainL))
    testLabels.append(np.array(testL))
    return trainData, np.array(trainLabels[0]), testData, np.array(testLabels[0])

def getDbfromCSV(path):
    files = os.listdir(path)
    DATA = np.zeros((3,40,2))
    for file in files:
        if file[-9:]=='moins.csv' and file[:7]=='sample1':
            X = open(path+file)
            for x, i in zip(X, range(20)):
                DATA[0, i] = np.array([float(coord) for coord in x.strip().split('\t')])
        if file[-9:]=='moins.csv' and file[:7]=='sample2':
            X = open(path+file)
            for x, i in zip(X, range(20)):
                DATA[1, i] = np.array([float(coord) for coord in x.strip().split('\t')])
        if file[-9:]=='moins.csv' and file[:7]=='sample3':
            X = open(path+file)
            for x, i in zip(X, range(20)):
                DATA[2, i] = np.array([float(coord) for coord in x.strip().split('\t')])

    for file in files:
        if file[-8:]=='plus.csv' and file[:7]=='sample1':
            print 'poulet'
            X = open(path+file)
            for x, i in zip(X, range(20)):
                DATA[0, i+20] = np.array([float(coord) for coord in x.strip().split('\t')])
        if file[-8:]=='plus.csv' and file[:7]=='sample2':
            X = open(path+file)
            for x, i in zip(X, range(20)):
                DATA[1, i+20] = np.array([float(coord) for coord in x.strip().split('\t')])
        if file[-8:]=='plus.csv' and file[:7]=='sample3':
            X = open(path+file)
            for x, i in zip(X, range(20)):
                DATA[2, i+20] = np.array([float(coord) for coord in x.strip().split('\t')])
    print DATA
    LABELS = np.zeros(40)
    LABELS[:20]=LABELS[:20]+1
    print LABELS
    # Y = np.genfromtxt(args.pathF + args.fileCL, delimiter=';')
    return DATA, LABELS

if __name__=='__main__':
    getDbfromCSV("/home/doob/OriginalData/")

# def equilibrateDataset(trainDataSet, trainLabels, pointedLabelIndice):
#     pointedClassIndices, notPointedClassesIndices, nbPointedExamples, nbNotPointedExamples = separateData(trainDataSet,
#                                                                                                           trainLabels,
#                                                                                                           pointedLabelIndice)
#     trainDataSet, trainLabels = selectData(trainDataSet, trainLabels, pointedClassIndices, notPointedClassesIndices,
#                                            nbPointedExamples, nbNotPointedExamples)
#     trainDataSetLength = len(trainDataSet)
#     shuffledIndices = np.arange(trainDataSetLength)
#     np.random.shuffle(shuffledIndices)
#     shuffledTrainDataSet = []
#     shuffledTrainLabels = []
#     for i in np.arange(trainDataSetLength):
#         shuffledTrainDataSet.append(trainDataSet[shuffledIndices[i]])
#         shuffledTrainLabels.append(trainLabels[shuffledIndices[i]])
#     return np.array(shuffledTrainDataSet), np.array(shuffledTrainLabels)