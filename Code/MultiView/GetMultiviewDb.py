import numpy as np
from string import digits
import os
import random
import logging
import h5py
import operator


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


def getFakeDB(features, pathF, name , NB_CLASS, LABELS_NAME):
    NB_VIEW = len(features)
    DATASET_LENGTH = int(pathF)
    VIEW_DIMENSIONS = np.random.random_integers(5, 20, NB_VIEW)

    DATA = dict((indx,
                        np.array([
                                     np.random.normal(0.0, 2, viewDimension)
                                     for i in np.arange(DATASET_LENGTH)]))
                        for indx, viewDimension in enumerate(VIEW_DIMENSIONS))

    CLASS_LABELS = np.random.random_integers(0, NB_CLASS-1, DATASET_LENGTH)
    LABELS_DICTIONARY = dict((indx, feature) for indx, feature in enumerate(features))
    return DATA, CLASS_LABELS, LABELS_DICTIONARY, DATASET_LENGTH


def getAwaLabels(nbLabels, pathToAwa):
    labelsFile = open(pathToAwa + 'Animals_with_Attributes/classes.txt', 'U')
    linesFile = [''.join(line.strip().split()).translate(None, digits) for line in labelsFile.readlines()]
    return linesFile


def getAwaDBcsv(views, pathToAwa, nameDB, nbLabels, LABELS_NAMES):
    awaLabels = getAwaLabels(nbLabels, pathToAwa)
    nbView = len(views)
    nbMaxLabels = len(awaLabels)
    if nbLabels == -1:
        nbLabels = nbMaxLabels
    nbNamesGiven = len(LABELS_NAMES)
    if nbNamesGiven > nbLabels:
        labelDictionary = {i:LABELS_NAMES[i] for i in np.arange(nbLabels)}
    elif nbNamesGiven < nbLabels and nbLabels <= nbMaxLabels:
        if LABELS_NAMES != ['']:
            labelDictionary = {i:LABELS_NAMES[i] for i in np.arange(nbNamesGiven)}
        else:
            labelDictionary = {}
            nbNamesGiven = 0
        nbLabelsToAdd = nbLabels-nbNamesGiven
        while nbLabelsToAdd > 0:
            currentLabel = random.choice(awaLabels)
            if currentLabel not in labelDictionary.values():
                labelDictionary[nbLabels-nbLabelsToAdd]=currentLabel
                nbLabelsToAdd -= 1
            else:
                pass
    else:
        labelDictionary = {i: LABELS_NAMES[i] for i in np.arange(nbNamesGiven)}
    viewDictionary = {i: views[i] for i in np.arange(nbView)}
    rawData = []
    labels = []
    nbExample = 0
    for view in np.arange(nbView):
        viewData = []
        for labelIndex in np.arange(nbLabels):
            pathToExamples = pathToAwa + 'Animals_with_Attributes/Features/' + viewDictionary[view] + '/' + \
                             labelDictionary[labelIndex] + '/'
            examples = os.listdir(pathToExamples)
            if view == 0:
                nbExample += len(examples)
            for example in examples:
                if viewDictionary[view]=='decaf':
                    exampleFile = open(pathToExamples + example)
                    viewData.append([float(line.strip()) for line in exampleFile])
                else:
                    exampleFile = open(pathToExamples + example)
                    viewData.append([[float(coordinate) for coordinate in raw.split()] for raw in exampleFile][0])
                if view == 0:
                    labels.append(labelIndex)

        rawData.append(np.array(viewData))
    data = rawData
    DATASET_LENGTH = len(labels)
    return data, labels, labelDictionary, DATASET_LENGTH


def getLabelSupports(CLASS_LABELS):
    labels = set(CLASS_LABELS)
    supports = [CLASS_LABELS.tolist().count(label) for label in labels]
    return supports, dict((label, index) for label, index in zip(labels, range(len(labels))))


def isUseful (labelSupports, index, CLASS_LABELS, labelDict):
    if labelSupports[labelDict[CLASS_LABELS[index]]] != 0:
        labelSupports[labelDict[CLASS_LABELS[index]]] -= 1
        return True, labelSupports
    else:
        return False, labelSupports


def splitDataset(DATASET, LEARNING_RATE, DATASET_LENGTH):
    LABELS = DATASET["/Labels/labelsArray"][...]
    NB_CLASS = int(DATASET["/nbClass"][...])
    validationIndices = extractRandomTrainingSet(LABELS, 1-LEARNING_RATE, DATASET_LENGTH, NB_CLASS)
    validationIndices.sort()
    return validationIndices


def extractRandomTrainingSet(CLASS_LABELS, LEARNING_RATE, DATASET_LENGTH, NB_CLASS):
    labelSupports, labelDict = getLabelSupports(np.array(CLASS_LABELS))
    nbTrainingExamples = [int(support * LEARNING_RATE) for support in labelSupports]
    trainingExamplesIndices = []
    usedIndices = []
    while nbTrainingExamples != [0 for i in range(NB_CLASS)]:
        isUseFull = False
        index = int(random.randint(0, DATASET_LENGTH-1))
        if index not in usedIndices:
            isUseFull, nbTrainingExamples = isUseful(nbTrainingExamples, index, CLASS_LABELS, labelDict)
        if isUseFull:
            trainingExamplesIndices.append(index)
            usedIndices.append(index)
    return trainingExamplesIndices


def getKFoldIndices(nbFolds, CLASS_LABELS, DATASET_LENGTH, NB_CLASS, learningIndices):
    labelSupports, labelDict = getLabelSupports(np.array(CLASS_LABELS[learningIndices]))
    nbTrainingExamples = [[int(support / nbFolds) for support in labelSupports] for fold in range(nbFolds)]
    trainingExamplesIndices = []
    usedIndices = []
    for foldIndex, fold in enumerate(nbTrainingExamples):
        trainingExamplesIndices.append([])
        while fold != [0 for i in range(NB_CLASS)]:
            index = random.randint(0, DATASET_LENGTH - 1)
            if learningIndices[index] not in usedIndices:
                isUseFull, fold = isUseful(fold, learningIndices[index], CLASS_LABELS, labelDict)
                if isUseFull:
                    trainingExamplesIndices[foldIndex].append(learningIndices[index])
                    usedIndices.append(learningIndices[index])
    return trainingExamplesIndices


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
    LABELS = np.zeros(40)
    LABELS[:20]=LABELS[:20]+1
    return DATA, LABELS


def getPositions(labelsUsed, fullLabels):
    usedIndices = []
    for labelIndex, label in enumerate(fullLabels):
        if label in labelsUsed:
            usedIndices.append(labelIndex)
    return usedIndices


def getClassicDBcsv(views, pathF, nameDB, NB_CLASS, LABELS_NAMES):
    fullDataset = []
    DATASET = h5py.File(nameDB+".hdf5", "w")
    fullLabels = np.genfromtxt(pathF + nameDB + '-ClassLabels.csv', delimiter=';').astype(int)
    if len(set(fullLabels))>NB_CLASS:
        labelsAvailable = list(set(fullLabels))
        labelsUsedIndices = np.random.randint(len(labelsAvailable), size=NB_CLASS)
        labelsUsed = labelsAvailable[labelsUsedIndices]
        usedIndices = getPositions(labelsUsed, fullLabels)
    else:
        labelsUsed = set(fullLabels)
        usedIndices = range(len(fullLabels))
    for viewIndex, view in enumerate(views):
        viewFile = pathF + nameDB + "-" + view + '.csv'
        viewMatrix = np.array(np.genfromtxt(viewFile, delimiter=';'))[usedIndices, :]
        DATASET["/View"+str(viewIndex)+"/matrix"] = viewMatrix
        DATASET["/View"+str(viewIndex)+"/name"] = view
        DATASET["/View"+str(viewIndex)+"/shape"] = viewMatrix.shape

    DATASET["/Labels/labelsArray"] = fullLabels[usedIndices]

    labelsNamesFile = open(pathF+nameDB+'-ClassLabels-Description.csv')
    labelsDictionary = dict((classIndice, labelName) for (classIndice, labelName) in [(int(line.strip().split(";")[0]),
                                                                                       line.strip().split(";")[1])
                                                                                      for lineIndex, line in labelsNamesFile if int(line.strip().split(";")[0]) in labelsUsed])
    DATASET["/datasetLength"] = len(DATASET["/Labels/labelsArray"][...])
    DATASET["/nbView"] = len(views)
    DATASET["/nbClass"] = len(set(DATASET["/Labels/labelsArray"][...]))
    # keptLabelsIndices = [labelIndice for labelIndice, labelName in labelsDictionary.items() if labelName in LABELS_NAMES]
    # maxNumbreOfClasses = len(labelsDictionary)
    #
    # if len(LABELS_NAMES) < NB_CLASS:
    #     classIndice = 0
    #     while classIndice < maxNumbreOfClasses:
    #         if classIndice not in keptLabelsIndices:
    #             keptLabelsIndices.append(classIndice)
    #         classIndice+=1
    #
    # elif len(LABELS_NAMES) > NB_CLASS:
    #     keptLabelsIndices = keptLabelsIndices[:NB_CLASS]
    #
    # DATASET = {}
    #
    # for featureIndex in range(len(fullDataset)):
    #     DATASET[featureIndex]=np.array([fullDataset[exampleIndice] for exampleIndice in range(datasetLength) if fullClasslabels[exampleIndice] in keptLabelsIndices])
    #
    # CLASS_LABELS = np.array([keptLabelsIndices.index(classLabel) for classLabel in fullClasslabels if classLabel in keptLabelsIndices])
    # DATASET_LENGTH = len(CLASS_LABELS)
    #
    # LABELS_DICTIONARY = dict((keptLabelsIndices.index(classLabel), labelsDictionary[classLabel]) for classLabel in keptLabelsIndices)

    return DATASET, labelsDictionary


def getCaltechDBcsv(views, pathF, nameDB, NB_CLASS, LABELS_NAMES):
    DATASET = h5py.File(nameDB+".hdf5", "w")
    fullLabels = np.genfromtxt(pathF + nameDB + '-ClassLabels.csv', delimiter=';').astype(int)
    if len(set(fullLabels))>NB_CLASS:
        labelsAvailable = list(set(fullLabels))
        labelsUsedIndices = np.random.randint(len(labelsAvailable), size=NB_CLASS)
        labelsUsed = labelsAvailable[labelsUsedIndices]
        usedIndices = getPositions(labelsUsed, fullLabels)
    else:
        labelsUsed = set(fullLabels)
        usedIndices = range(len(fullLabels))
    for viewIndex, view in enumerate(views):
        viewFile = pathF + nameDB + "-" + view + '.csv'
        viewMatrix = np.array(np.genfromtxt(viewFile, delimiter=';'))[usedIndices, :]
        DATASET["/View"+str(viewIndex)+"/matrix"] = viewMatrix
        DATASET["/View"+str(viewIndex)+"/name"] = view
        DATASET["/View"+str(viewIndex)+"/shape"] = viewMatrix.shape

    DATASET["/Labels/labelsArray"] = fullLabels[usedIndices]

    labelsNamesFile = open(pathF+nameDB+'-ClassLabels-Description.csv')
    labelsDictionary = dict((classIndice, labelName) for (classIndice, labelName) in [(int(line.strip().split(";")[0]),
                                                                                       line.strip().split(";")[1])
                                                                                      for lineIndex, line in labelsNamesFile if int(line.strip().split(";")[0]) in labelsUsed])
    DATASET["/datasetLength"] = len(DATASET["/Labels/labelsArray"][...])
    DATASET["/nbView"] = len(views)
    DATASET["/nbClass"] = len(set(DATASET["/Labels/labelsArray"][...]))
    # keptLabelsIndices = [labelIndice for labelIndice, labelName in labelsDictionary.items() if labelName in LABELS_NAMES]
    # maxNumbreOfClasses = len(labelsDictionary)
    #
    # if len(LABELS_NAMES) < NB_CLASS:
    #     classIndice = 0
    #     while classIndice < maxNumbreOfClasses:
    #         if classIndice not in keptLabelsIndices:
    #             keptLabelsIndices.append(classIndice)
    #         classIndice+=1
    #
    # elif len(LABELS_NAMES) > NB_CLASS:
    #     keptLabelsIndices = keptLabelsIndices[:NB_CLASS]
    #
    # DATASET = {}
    #
    # for featureIndex in range(len(fullDataset)):
    #     DATASET[featureIndex]=np.array([fullDataset[exampleIndice] for exampleIndice in range(datasetLength) if fullClasslabels[exampleIndice] in keptLabelsIndices])
    #
    # CLASS_LABELS = np.array([keptLabelsIndices.index(classLabel) for classLabel in fullClasslabels if classLabel in keptLabelsIndices])
    # DATASET_LENGTH = len(CLASS_LABELS)
    #
    # LABELS_DICTIONARY = dict((keptLabelsIndices.index(classLabel), labelsDictionary[classLabel]) for classLabel in keptLabelsIndices)

    return DATASET, labelsDictionary


def getMultiOmicDBcsv(features, path, name, NB_CLASS, LABELS_NAMES):

    datasetFile = h5py.File(path+"MultiOmicDataset.hdf5", "w")

    logging.debug("Start:\t Getting Methylation Data")
    methylData = np.genfromtxt(path+"matching_methyl.csv", delimiter=',')
    datasetFile["/View0/matrix"] = methylData
    datasetFile["/View0/name"] = "Methyl"
    datasetFile["/View0/shape"] = methylData.shape
    logging.debug("Done:\t Getting Methylation Data")

    logging.debug("Start:\t Getting MiRNA Data")
    mirnaData = np.genfromtxt(path+"matching_mirna.csv", delimiter=',')
    datasetFile["/View1/matrix"] = mirnaData
    datasetFile["/View1/name"] = "MiRNA_"
    datasetFile["/View1/shape"] = mirnaData.shape
    logging.debug("Done:\t Getting MiRNA Data")

    logging.debug("Start:\t Getting RNASeq Data")
    rnaseqData = np.genfromtxt(path+"matching_rnaseq.csv", delimiter=',')
    datasetFile["/View2/matrix"] = rnaseqData
    datasetFile["/View2/name"] = "RNASeq"
    datasetFile["/View2/shape"] = rnaseqData.shape
    logging.debug("Done:\t Getting RNASeq Data")

    logging.debug("Start:\t Getting Clinical Data")
    clinical = np.genfromtxt(path+"clinicalMatrix.csv", delimiter=',')
    datasetFile["/View3/matrix"] = clinical
    datasetFile["/View3/name"] = "Clinic"
    datasetFile["/View3/shape"] = clinical.shape
    logging.debug("Done:\t Getting Clinical Data")

    labelFile = open(path+'brca_labels_triple-negatif.csv')
    LABELS = np.array([int(line.strip().split(',')[1]) for line in labelFile])
    datasetFile["/Labels/labelsArray"] = LABELS

    datasetFile["/nbView"] = 4
    datasetFile["/nbClass"] = 2
    datasetFile["/datasetLength"] = len(datasetFile["/Labels/labelsArray"])
    labelDictionary = {0:"No", 1:"Yes"}
    # datasetFile = getPseudoRNASeq(datasetFile)
    return datasetFile, labelDictionary

def getMultiOmicModifiedDBcsv(features, path, name, NB_CLASS, LABELS_NAMES):

    datasetFile = h5py.File(path+"ModifiedMultiOmicDataset.hdf5", "w")

    logging.debug("Start:\t Getting Methylation Data")
    methylData = np.genfromtxt(path+"matching_methyl.csv", delimiter=',')
    datasetFile["/View0/matrix"] = methylData
    datasetFile["/View0/name"] = "Methyl_"
    datasetFile["/View0/shape"] = methylData.shape
    logging.debug("Done:\t Getting Methylation Data")

    logging.debug("Start:\t Getting MiRNA Data")
    mirnaData = np.genfromtxt(path+"matching_mirna.csv", delimiter=',')
    datasetFile["/View1/matrix"] = mirnaData
    datasetFile["/View1/name"] = "MiRNA__"
    datasetFile["/View1/shape"] = mirnaData.shape
    logging.debug("Done:\t Getting MiRNA Data")

    logging.debug("Start:\t Getting RNASeq Data")
    rnaseqData = np.genfromtxt(path+"matching_rnaseq.csv", delimiter=',')
    datasetFile["/View2/matrix"] = rnaseqData
    datasetFile["/View2/name"] = "RNASeq_"
    datasetFile["/View2/shape"] = rnaseqData.shape
    logging.debug("Done:\t Getting RNASeq Data")

    logging.debug("Start:\t Getting Clinical Data")
    clinical = np.genfromtxt(path+"clinicalMatrix.csv", delimiter=',')
    datasetFile["/View3/matrix"] = clinical
    datasetFile["/View3/name"] = "Clinic_"
    datasetFile["/View3/shape"] = clinical.shape
    logging.debug("Done:\t Getting Clinical Data")

    logging.debug("Start:\t Getting Labels")
    labelFile = open(path+'brca_labels_triple-negatif.csv')
    LABELS = np.array([int(line.strip().split(',')[1]) for line in labelFile])
    datasetFile["/Labels/labelsArray"] = LABELS
    logging.debug("Done:\t Getting Labels")

    logging.debug("Start:\t Getting Data Shape")
    datasetFile["/nbView"] = 5
    datasetFile["/nbClass"] = 2
    datasetFile["/datasetLength"] = len(datasetFile["/Labels/labelsArray"])
    labelDictionary = {0:"No", 1:"Yes"}
    logging.debug("Done:\t Getting Data Shape")

    logging.debug("Start:\t Getting Modified RNASeq Data")
    RNASeq = datasetFile["View2/matrix"][...]
    modifiedRNASeq = np.zeros((datasetFile.get("datasetLength/").value, datasetFile["View2/shape"][1]), dtype=int)
    for exampleindice, exampleArray in enumerate(RNASeq):
        RNASeqDictionary = dict((index, value) for index, value in enumerate(exampleArray))
        sorted_x = sorted(RNASeqDictionary.items(), key=operator.itemgetter(1))
        modifiedRNASeq[exampleindice] = np.array([index for (index, value) in sorted_x], dtype=int)
    datasetFile["/View4/matrix"] = modifiedRNASeq
    datasetFile["/View4/name"] = "MRNASeq"
    datasetFile["/View4/shape"] = modifiedRNASeq.shape
    logging.debug("Done:\t Getting Modified RNASeq Data")

    return datasetFile, labelDictionary


def getMultiOmicModifiedDBhdf5(features, path, name, NB_CLASS, LABELS_NAMES):
    datasetFile = h5py.File(path+"ModifiedMultiOmicDataset.hdf5", "r")
    labelDictionary = {0:"No", 1:"Yes"}
    return datasetFile, labelDictionary


def makeArrayFromTriangular(pseudoRNASeqMatrix):
    matrixShape = len(pseudoRNASeqMatrix[0,:])
    exampleArray = np.array(((matrixShape-1)*matrixShape)/2)
    arrayIndex = 0
    for i in range(matrixShape-1):
        for j in range(i+1, matrixShape):
            exampleArray[arrayIndex]=pseudoRNASeqMatrix[i,j]
            arrayIndex += 1
    return exampleArray


def getPseudoRNASeq(dataset):
    nbGenes = len(dataset["/View2/matrix"][0, :])
    pseudoRNASeq = np.zeros((dataset["/datasetlength"][...], ((nbGenes - 1) * nbGenes) / 2), dtype=bool_)
    for exampleIndex in xrange(dataset["/datasetlength"][...]):
        arrayIndex = 0
        for i in xrange(nbGenes):
            for j in xrange(nbGenes):
                if i > j:
                    pseudoRNASeq[exampleIndex, arrayIndex] = dataset["/View2/matrix"][exampleIndex, j] < dataset["/View2/matrix"][exampleIndex, i]
                    arrayIndex += 1
    dataset["/View4/matrix"] = pseudoRNASeq
    dataset["/View4/name"] = "pseudoRNASeq"
    return dataset


def getMultiOmicModDBhdf5(features, path, name, NB_CLASS, LABELS_NAMES):
    datasetFile = h5py.File(path+"MultiOmicDataset.hdf5", "r")
    labelDictionary = {0:"No", 1:"Yes"}
    return datasetFile, labelDictionary





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
