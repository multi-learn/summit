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


def getFakeDBhdf5(features, pathF, name , NB_CLASS, LABELS_NAME):
    NB_VIEW = len(features)
    DATASET_LENGTH = 300
    NB_CLASS = 2
    VIEW_DIMENSIONS = np.random.random_integers(5, 20, NB_VIEW)

    DATA = dict((indx,
                        np.array([
                                     np.random.normal(0.0, 2, viewDimension)
                                     for i in np.arange(DATASET_LENGTH)]))
                        for indx, viewDimension in enumerate(VIEW_DIMENSIONS))

    CLASS_LABELS = np.random.random_integers(0, NB_CLASS-1, DATASET_LENGTH)
    print CLASS_LABELS
    LABELS_DICTIONARY = dict((indx, feature) for indx, feature in enumerate(features))
    datasetFile = h5py.File(pathF+"Fake.hdf5", "w")
    for index, viewData in enumerate(DATA.values()):
        viewDset = datasetFile.create_dataset("View"+str(index), viewData.shape)
        viewDset[...] = viewData
        viewDset.attrs["name"] = "View"+str(index)
    labelsDset = datasetFile.create_dataset("labels", CLASS_LABELS.shape)
    labelsDset[...] = CLASS_LABELS
    labelsDset.attrs["name"] = "Labels"

    metaDataGrp = datasetFile.create_group("Metadata")
    metaDataGrp.attrs["nbView"] = NB_VIEW
    metaDataGrp.attrs["nbClass"] = NB_CLASS
    metaDataGrp.attrs["datasetLength"] = len(CLASS_LABELS)
    labelDictionary = {0:"No", 1:"Yes"}
    datasetFile.close()
    datasetFile = h5py.File(pathF+"Fake.hdf5", "r")
    return datasetFile, LABELS_DICTIONARY


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
    LABELS = DATASET.get("labels")[...]
    NB_CLASS = int(DATASET["Metadata"].attrs["nbClass"])
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


def getKFoldIndices(nbFolds, CLASS_LABELS, NB_CLASS, learningIndices):
    labelSupports, labelDict = getLabelSupports(np.array(CLASS_LABELS[learningIndices]))
    nbTrainingExamples = [[int(support / nbFolds) for support in labelSupports] for fold in range(nbFolds)]
    trainingExamplesIndices = []
    usedIndices = []
    for foldIndex, fold in enumerate(nbTrainingExamples):
        trainingExamplesIndices.append([])
        while fold != [0 for i in range(NB_CLASS)]:
            index = random.randint(0, len(learningIndices)-1)
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
    datasetFile = h5py.File(nameDB+".hdf5", "w")
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
        datasetFile["/View"+str(viewIndex)+"/matrix"] = viewMatrix
        datasetFile["/View"+str(viewIndex)+"/name"] = view
        datasetFile["/View"+str(viewIndex)+"/shape"] = viewMatrix.shape

    datasetFile["/Labels/labelsArray"] = fullLabels[usedIndices]

    labelsNamesFile = open(pathF+nameDB+'-ClassLabels-Description.csv')
    labelsDictionary = dict((classIndice, labelName) for (classIndice, labelName) in [(int(line.strip().split(";")[0]),
                                                                                       line.strip().split(";")[1])
                                                                                      for lineIndex, line in labelsNamesFile if int(line.strip().split(";")[0]) in labelsUsed])
    datasetFile["/datasetLength"] = len(datasetFile["/Labels/labelsArray"][...])
    datasetFile["/nbView"] = len(views)
    datasetFile["/nbClass"] = len(set(datasetFile["/Labels/labelsArray"][...]))
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
    # datasetFile = {}
    #
    # for featureIndex in range(len(fullDataset)):
    #     datasetFile[featureIndex]=np.array([fullDataset[exampleIndice] for exampleIndice in range(datasetLength) if fullClasslabels[exampleIndice] in keptLabelsIndices])
    #
    # CLASS_LABELS = np.array([keptLabelsIndices.index(classLabel) for classLabel in fullClasslabels if classLabel in keptLabelsIndices])
    # DATASET_LENGTH = len(CLASS_LABELS)
    #
    # LABELS_DICTIONARY = dict((keptLabelsIndices.index(classLabel), labelsDictionary[classLabel]) for classLabel in keptLabelsIndices)

    return datasetFile, labelsDictionary


def getMultiOmicDBcsv(features, path, name, NB_CLASS, LABELS_NAMES):

    datasetFile = h5py.File(path+"MultiOmic.hdf5", "w")

    logging.debug("Start:\t Getting Methylation Data")
    methylData = np.genfromtxt(path+"matching_methyl.csv", delimiter=',')
    methylDset = datasetFile.create_dataset("View0", methylData.shape)
    methylDset[...] = methylData
    methylDset.attrs["name"] = "Methyl"
    logging.debug("Done:\t Getting Methylation Data")

    logging.debug("Start:\t Getting MiRNA Data")
    mirnaData = np.genfromtxt(path+"matching_mirna.csv", delimiter=',')
    mirnaDset = datasetFile.create_dataset("View1", mirnaData.shape)
    mirnaDset[...] = mirnaData
    mirnaDset.attrs["name"]="MiRNA_"
    logging.debug("Done:\t Getting MiRNA Data")

    logging.debug("Start:\t Getting RNASeq Data")
    rnaseqData = np.genfromtxt(path+"matching_rnaseq.csv", delimiter=',')
    rnaseqDset = datasetFile.create_dataset("View2", rnaseqData.shape)
    rnaseqDset[...] = rnaseqData
    rnaseqDset.attrs["name"]="RNASeq"
    logging.debug("Done:\t Getting RNASeq Data")

    logging.debug("Start:\t Getting Clinical Data")
    clinical = np.genfromtxt(path+"clinicalMatrix.csv", delimiter=',')
    clinicalDset = datasetFile.create_dataset("View3", clinical.shape)
    clinicalDset[...] = clinical
    clinicalDset.attrs["name"] = "Clinic"
    logging.debug("Done:\t Getting Clinical Data")

    labelFile = open(path+'brca_labels_triple-negatif.csv')
    labels = np.array([int(line.strip().split(',')[1]) for line in labelFile])
    labelsDset = datasetFile.create_dataset("labels", labels.shape)
    labelsDset[...] = labels
    labelsDset.attrs["name"] = "Labels"

    metaDataGrp = datasetFile.create_group("Metadata")
    metaDataGrp.attrs["nbView"] = 4
    metaDataGrp.attrs["nbClass"] = 2
    metaDataGrp.attrs["datasetLength"] = len(labels)
    labelDictionary = {0:"No", 1:"Yes"}
    datasetFile.close()
    datasetFile = h5py.File(path+"MultiOmic.hdf5", "r")
    # datasetFile = getPseudoRNASeq(datasetFile)
    return datasetFile, labelDictionary


def getModifiedMultiOmicDBcsv(features, path, name, NB_CLASS, LABELS_NAMES):

    datasetFile = h5py.File(path+"ModifiedMultiOmic.hdf5", "w")

    logging.debug("Start:\t Getting Methylation Data")
    methylData = np.genfromtxt(path+"matching_methyl.csv", delimiter=',')
    methylDset = datasetFile.create_dataset("View0", methylData.shape)
    methylDset[...] = methylData
    methylDset.attrs["name"] = "Methyl_"
    logging.debug("Done:\t Getting Methylation Data")

    logging.debug("Start:\t Getting MiRNA Data")
    mirnaData = np.genfromtxt(path+"matching_mirna.csv", delimiter=',')
    mirnaDset = datasetFile.create_dataset("View1", mirnaData.shape)
    mirnaDset[...] = mirnaData
    mirnaDset.attrs["name"]="MiRNA__"
    logging.debug("Done:\t Getting MiRNA Data")

    logging.debug("Start:\t Getting RNASeq Data")
    rnaseqData = np.genfromtxt(path+"matching_rnaseq.csv", delimiter=',')
    uselessRows = []
    for rowIndex, row in enumerate(np.transpose(rnaseqData)):
        if not row.any():
            uselessRows.append(rowIndex)
    usefulRows = [usefulRowIndex for usefulRowIndex in range(rnaseqData.shape[1]) if usefulRowIndex not in uselessRows]
    rnaseqDset = datasetFile.create_dataset("View2", (rnaseqData.shape[0], len(usefulRows)))
    rnaseqDset[...] = rnaseqData[:, usefulRows]
    rnaseqDset.attrs["name"]="RNASeq_"
    logging.debug("Done:\t Getting RNASeq Data")

    logging.debug("Start:\t Getting Clinical Data")
    clinical = np.genfromtxt(path+"clinicalMatrix.csv", delimiter=',')
    clinicalDset = datasetFile.create_dataset("View3", clinical.shape)
    clinicalDset[...] = clinical
    clinicalDset.attrs["name"] = "Clinic_"
    logging.debug("Done:\t Getting Clinical Data")



    logging.debug("Start:\t Getting Sorted RNASeq Data")
    RNASeq = datasetFile["View2"][...]
    modifiedRNASeq = np.zeros((datasetFile.get("Metadata").attrs["datasetLength"], datasetFile.get("View2").shape[1]), dtype=int)
    for exampleIndex, exampleArray in enumerate(RNASeq):
        RNASeqDictionary = dict((index, value) for index, value in enumerate(exampleArray))
        sorted_x = sorted(RNASeqDictionary.items(), key=operator.itemgetter(1))
        modifiedRNASeq[exampleIndex] = np.array([index for (index, value) in sorted_x], dtype=int)
    mrnaseqDset = datasetFile.create_dataset("View4", modifiedRNASeq.shape, data=modifiedRNASeq)
    mrnaseqDset.attrs["name"] = "SRNASeq"
    logging.debug("Done:\t Getting Sorted RNASeq Data")

    logging.debug("Start:\t Getting Binned RNASeq Data")
    SRNASeq = datasetFile["View4"][...]
    nbBins = 372
    binLen = 935
    binOffset = 187
    bins = np.zeros((nbBins, binLen), dtype=int)
    for binIndex in range(nbBins):
        bins[binIndex] = np.arange(binLen)+binIndex*binOffset
    binnedRNASeq = np.zeros((datasetFile.get("Metadata").attrs["datasetLength"], datasetFile.get("View2").shape[1]*nbBins), dtype=bool)
    for exampleIndex, exampleArray in enumerate(SRNASeq):
        for geneRank, geneIndex in enumerate(exampleArray):
            for binIndex, bin in bins:
                if geneRank in bin:
                    binnedRNASeq[exampleIndex, geneIndex*nbBins+binIndex] = True
    brnaseqDset = datasetFile.create_dataset("View5", binnedRNASeq.shape, data=binnedRNASeq)
    brnaseqDset.attrs["name"] = "BRNASeq"
    logging.debug("Done:\t Getting Binned RNASeq Data")

    labelFile = open(path+'brca_labels_triple-negatif.csv')
    labels = np.array([int(line.strip().split(',')[1]) for line in labelFile])
    labelsDset = datasetFile.create_dataset("labels", labels.shape)
    labelsDset[...] = labels
    labelsDset.attrs["name"] = "Labels"

    metaDataGrp = datasetFile.create_group("Metadata")
    metaDataGrp.attrs["nbView"] = 5
    metaDataGrp.attrs["nbClass"] = 2
    metaDataGrp.attrs["datasetLength"] = len(labels)
    labelDictionary = {0:"No", 1:"Yes"}
    # datasetFile = h5py.File(path+"ModifiedMultiOmic.hdf5", "r")
    # logging.debug("Start:\t Getting Binary RNASeq Data")
    # binarizedRNASeqDset = datasetFile.create_dataset("View5", shape=(len(labels), len(rnaseqData)*(len(rnaseqData)-1)/2), dtype=bool)
    # for exampleIndex in range(len(labels)):
    #     offseti=0
    #     rnaseqData = datasetFile["View2"][exampleIndex]
    #     for i, idata in enumerate(rnaseqData):
    #         for j, jdata in enumerate(rnaseqData):
    #             if i < j:
    #                 binarizedRNASeqDset[offseti+j] = idata > jdata
    #         offseti += len(rnaseqData)-i-1
    # binarizedRNASeqDset.attrs["name"] = "BRNASeq"
    # i=0
    # for featureIndex in range(len(rnaseqData)*(len(rnaseqData)-1)/2):
    #     if allSame(binarizedRNASeqDset[:, featureIndex]):
    #         i+=1
    # print i
    # logging.debug("Done:\t Getting Binary RNASeq Data")


    datasetFile.close()
    datasetFile = h5py.File(path+"ModifiedMultiOmic.hdf5", "r")

    return datasetFile, labelDictionary


def allSame(array):
    value = array[0]
    areAllSame = True
    for i in array:
        if i != value:
            areAllSame = False
    return areAllSame


def getModifiedMultiOmicDBhdf5(features, path, name, NB_CLASS, LABELS_NAMES):
    datasetFile = h5py.File(path+"ModifiedMultiOmic.hdf5", "r")
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


def getMultiOmicDBhdf5(features, path, name, NB_CLASS, LABELS_NAMES):
    datasetFile = h5py.File(path+"MultiOmic.hdf5", "r")
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
