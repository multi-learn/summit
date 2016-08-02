import numpy as np
from string import digits
import os
import random


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
    return linesFile


def getAwaDB(views, pathToAwa, nameDB, nbLabels, LABELS_NAMES):
    awaLabels = getAwaLabels(nbLabels, pathToAwa)
    nbView = len(views)
    nbMaxLabels = len(awaLabels)
    if nbLabels == -1:
        nbLabels = nbMaxLabels
    nbNamesGiven = len(LABELS_NAMES)
    if nbNamesGiven > nbLabels:
        labelDictionnary = {i:LABELS_NAMES[i] for i in np.arange(nbLabels)}
    elif nbNamesGiven < nbLabels and nbLabels <= nbMaxLabels:
        if LABELS_NAMES != ['']:
            labelDictionnary = {i:LABELS_NAMES[i] for i in np.arange(nbNamesGiven)}
        else:
            labelDictionnary = {}
            nbNamesGiven = 0
        nbLabelsToAdd = nbLabels-nbNamesGiven
        while nbLabelsToAdd > 0:
            currentLabel = random.choice(awaLabels)
            if currentLabel not in labelDictionnary.values():
                labelDictionnary[nbLabels-nbLabelsToAdd]=currentLabel
                nbLabelsToAdd -= 1
            else:
                pass
    else:
        labelDictionnary = {i: LABELS_NAMES[i] for i in np.arange(nbNamesGiven)}
    viewDictionnary = {i: views[i] for i in np.arange(nbView)}
    rawData = []
    labels = []
    nbExample = 0
    # ij = []
    for view in np.arange(nbView):
        viewData = []
        for labelIndex in np.arange(nbLabels):
            pathToExamples = pathToAwa + 'Animals_with_Attributes/Features/' + viewDictionnary[view] + '/' + \
                             labelDictionnary[labelIndex] + '/'
            examples = os.listdir(pathToExamples)
            if view == 0:
                nbExample += len(examples)
            for example in examples:
                if viewDictionnary[view]=='decaf':
                    exampleFile = open(pathToExamples + example)
                    viewData.append([float(line.strip()) for line in exampleFile])
                else:
                    exampleFile = open(pathToExamples + example)
                    viewData.append([[float(coordinate) for coordinate in raw.split()] for raw in exampleFile][0])
                if view == 0:
                    labels.append(labelIndex)

        rawData.append(np.array(viewData))
    data = rawData
    # data = np.empty((nbExample, nbView), dtype=list)
    # for viewIdice in np.arange(nbView):
    #     for exampleIndice in np.arange(nbExample):
    #         data[exampleIndice, viewIdice] = rawData[viewIdice][exampleIndice]
    #         # data[exampleIndice, viewIdice] = {i:rawData[viewIdice][exampleIndice][i] for i in np.arange(len(rawData[viewIdice][exampleIndice]))}
    DATASET_LENGTH = len(labels)
    return data, labels, labelDictionnary, DATASET_LENGTH


def getLabelSupports (CLASS_LABELS):
    labels = set(CLASS_LABELS)
    supports = [CLASS_LABELS.tolist().count(label) for label in labels]
    return supports, dict((label, index) for label, index in zip(labels, range(len(labels))))


def isUseful (labelSupports, index, CLASS_LABELS, labelDict):
    if labelSupports[labelDict[CLASS_LABELS[index]]] != 0:
        labelSupports[labelDict[CLASS_LABELS[index]]] -= 1
        return True, labelSupports
    else :
        return False, labelSupports


def extractRandomTrainingSet(DATA, CLASS_LABELS, LEARNING_RATE, DATASET_LENGTH, NB_VIEW, NB_CLASS):
    labelSupports, labelDict = getLabelSupports (CLASS_LABELS)
    nbTrainingExamples = [int(support * LEARNING_RATE) for support in labelSupports]
    trainingExamplesIndices = []
    while nbTrainingExamples != [0 for i in range(NB_CLASS)]:
        index = random.randint(0, DATASET_LENGTH-1)
        isUseFull, nbTrainingExamples = isUseful(nbTrainingExamples, index, CLASS_LABELS, labelDict)
        if isUseFull:
            trainingExamplesIndices.append(index)

    # trainingExamplesIndices = np.random.random_integers(0, DATASET_LENGTH-1, nbTrainingExamples)
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


def getKFoldIndices(nbFolds, CLASS_LABELS, DATASET_LENGTH, NB_CLASS):
    labelSupports, labelDict = getLabelSupports(CLASS_LABELS)
    nbTrainingExamples = [[int(support / nbFolds) for fold in range(nbFolds)] for support in labelSupports]
    trainingExamplesIndices = []
    usedIndices = []
    for foldIndex, fold in enumerate(nbTrainingExamples):
        trainingExamplesIndices.append([])
        while fold != [0 for i in range(NB_CLASS)]:
            index = random.randint(0, DATASET_LENGTH - 1)
            if index not in usedIndices:
                isUseFull, fold = isUseful(fold, index, CLASS_LABELS, labelDict)
                if isUseFull:
                    trainingExamplesIndices[foldIndex].append(index)
                    usedIndices.append(index)


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


def getCaltechDB(features, pathF, nameDB, NB_CLASS, LABELS_NAMES):
    fullDataset = []
    for feature in features:
        featureFile = pathF + nameDB + "-" + feature + '.csv'
        fullDataset.append(np.genfromtxt(featureFile, delimiter=';'))

    fullClasslabels = np.genfromtxt(pathF + nameDB + '-ClassLabels.csv', delimiter=';').astype(int)

    labelsNamesFile = open(pathF+nameDB+'-ClassLabels-Description.csv')
    labelsDictionary = dict((classIndice, labelName) for (classIndice, labelName) in [(int(line.split().strip(";")[0]), line.split().strip(";")[1]) for line in labelsNamesFile])

    datasetLength = len(fullClasslabels)

    keptLabelsIndices = [labelIndice for labelIndice, labelName in labelsDictionary.items() if labelName in LABELS_NAMES]
    maxNumbreOfClasses = len(labelsDictionary)

    if len(LABELS_NAMES) < NB_CLASS:
        classIndice = 0
        while classIndice < maxNumbreOfClasses:
            if classIndice not in keptLabelsIndices:
                keptLabelsIndices.append(classIndice)
            classIndice+=1

    elif len(LABELS_NAMES) > NB_CLASS:
        keptLabelsIndices = keptLabelsIndices[:NB_CLASS]

    DATASET = {}

    for featureIndice in range(len(fullDataset)):
        DATASET[featureIndice]=np.array([fullDataset[exampleIndice] for exampleIndice in range(datasetLength) if fullClasslabels[exampleIndice] in keptLabelsIndices])

    CLASS_LABELS = np.array([keptLabelsIndices.index(classLabel) for classLabel in fullClasslabels if classLabel in keptLabelsIndices])
    DATASET_LENGTH = len(CLASS_LABELS)

    LABELS_DICTIONARY = dict((keptLabelsIndices.index(classLabel), labelsDictionary[classLabel]) for classLabel in keptLabelsIndices)

    return DATASET, CLASS_LABELS, LABELS_DICTIONARY, DATASET_LENGTH

def getMultiOmicDB(features, path, name, NB_CLASS, LABELS_NAMES):
    methylData = np.genfromtxt(path+"matching_methyl.csv", delimiter=',')
    mirnaData = np.genfromtxt(path+"matching_mirna.csv", delimiter=',')
    rnaseqData = np.genfromtxt(path+"matching_rnaseq.csv", delimiter=',')
    clinical = np.genfromtxt(path+"clinicalMatrix.csv", delimiter=',')
    DATASET = {0:methylData, 1:mirnaData, 2:rnaseqData, 3:clinical}
    DATASET_LENGTH = len(methylData)
    labelFile = open(path+'brca_labels_triple-negatif.csv')
    CLASS_LABELS = np.array([int(line.strip().split(',')[1]) for line in labelFile])
    labelDictionnary = {0:"No", 1:"Yes"}
    return DATASET, CLASS_LABELS, labelDictionnary, DATASET_LENGTH






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
