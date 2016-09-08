import numpy as np
import math
from scipy import sparse, io
from string import digits
import os
import random
import logging
import h5py
import operator

# Author-Info
__author__ 	= "Baptiste Bauvin"
__status__ 	= "Prototype"                           # Production, Development, Prototype


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
    LABELS_DICTIONARY = dict((indx, feature) for indx, feature in enumerate(features))
    datasetFile = h5py.File(pathF+"Fake.hdf5", "w")
    for index, viewData in enumerate(DATA.values()):
        if index == 0:
            viewData = sparse.csr_matrix(viewData)
            viewGrp = datasetFile.create_group("View0")
            dataDset = viewGrp.create_dataset("data", viewData.data.shape, data=viewData.data)
            indicesDset = viewGrp.create_dataset("indices", viewData.indices.shape, data=viewData.indices)
            indptrDset = viewGrp.create_dataset("indptr", viewData.indptr.shape, data=viewData.indptr)
            viewGrp.attrs["name"] = "View"+str(index)
            viewGrp.attrs["sparse"] = True
            viewGrp.attrs["shape"] = viewData.shape
        else:
            viewDset = datasetFile.create_dataset("View"+str(index), viewData.shape)
            viewDset[...] = viewData
            viewDset.attrs["name"] = "View"+str(index)
            viewDset.attrs["sparse"] = False
    labelsDset = datasetFile.create_dataset("Labels", CLASS_LABELS.shape)
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
    LABELS = DATASET.get("Labels")[...]
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


def getPositions(labelsUsed, fullLabels):
    usedIndices = []
    for labelIndex, label in enumerate(fullLabels):
        if label in labelsUsed:
            usedIndices.append(labelIndex)
    return usedIndices


def getClassicDBcsv(views, pathF, nameDB, NB_CLASS, LABELS_NAMES):
    datasetFile = h5py.File(pathF+nameDB+".hdf5", "w")
    labelsNamesFile = open(pathF+nameDB+'-ClassLabels-Description.csv')
    if len(LABELS_NAMES)!=NB_CLASS:
        nbLabelsAvailable = 0
        for l in labelsNamesFile:
            nbLabelsAvailable+=1
        LABELS_NAMES = [line.strip().split(";")[1] for lineIdx, line in enumerate(labelsNamesFile) if lineIdx in np.random.randint(nbLabelsAvailable, size=NB_CLASS)]
    fullLabels = np.genfromtxt(pathF + nameDB + '-ClassLabels.csv', delimiter=';').astype(int)
    labelsDictionary = dict((classIndice, labelName) for (classIndice, labelName) in
                        [(int(line.strip().split(";")[0]),line.strip().split(";")[1])for lineIndex, line in labelsNamesFile if line.strip().split(";")[0] in LABELS_NAMES])
    if len(set(fullLabels))>NB_CLASS:
        usedIndices = getPositions(labelsDictionary.keys(), fullLabels)
    else:
        usedIndices = range(len(fullLabels))
    for viewIndex, view in enumerate(views):
        viewFile = pathF + nameDB + "-" + view + '.csv'
        viewMatrix = np.array(np.genfromtxt(viewFile, delimiter=';'))[usedIndices, :]
        viewDset = datasetFile.create_dataset("View"+str(viewIndex), viewMatrix.shape, data=viewMatrix)
        viewDset.attrs["name"] = view

    labelsDset = datasetFile.create_dataset("Labels", fullLabels[usedIndices].shape, data=fullLabels[usedIndices])
    labelsDset.attrs["labelsDictionary"] = labelsDictionary

    metaDataGrp = datasetFile.create_group("Metadata")
    metaDataGrp.attrs["nbView"] = len(views)
    metaDataGrp.attrs["nbClass"] = NB_CLASS
    metaDataGrp.attrs["datasetLength"] = len(fullLabels[usedIndices])
    datasetFile.close()
    datasetFile = h5py.File(pathF+nameDB+".hdf5", "r")
    return datasetFile, labelsDictionary


def getClassicDBhdf5(views, pathF, nameDB, NB_CLASS, LABELS_NAMES):
    datasetFile = h5py.File(pathF+nameDB+".hdf5", "r")
    fullLabels = datasetFile.get("Labels").value
    fullLabelsDictionary = datasetFile.get("Labels").attrs["labelsDictionary"]
    fullNbClass = datasetFile.get("Metadata").attrs["nbClass"]
    if len(LABELS_NAMES)!=NB_CLASS:
        LABELS_NAMES = [value for index, value in fullLabelsDictionary.iteritems()
                        if index in np.random.randint(fullNbClass, size=NB_CLASS)]
    labelsDictionary = dict((classIndice, labelName) for (classIndice, labelName)
                            in fullLabelsDictionary.iteritems() if labelName in LABELS_NAMES)
    if len(set(fullLabels))>NB_CLASS:
        usedIndices = getPositions(labelsDictionary.keys(), fullLabels)
    else:
        usedIndices = range(len(fullLabels))
    tempDatasetFile = datasetFile = h5py.File(pathF+nameDB+"_temp.hdf5", "w")
    for viewIndex, view in enumerate(views):
        viewMatrix = datasetFile.get("View"+str(viewIndex)).value[:, usedIndices]
        viewDset = tempDatasetFile.create_dataset("View"+str(viewIndex), viewMatrix.shape, data=viewMatrix)
        viewDset.attrs["name"] = view

    labelsDset = tempDatasetFile.create_dataset("Labels", fullLabels[usedIndices].shape, data=fullLabels[usedIndices])
    labelsDset.attrs["labelsDictionary"] = labelsDictionary

    metaDataGrp = tempDatasetFile.create_group("Metadata")
    metaDataGrp.attrs["nbView"] = len(views)
    metaDataGrp.attrs["nbClass"] = NB_CLASS
    metaDataGrp.attrs["datasetLength"] = len(fullLabels[usedIndices])
    datasetFile.close()
    tempDatasetFile.close()
    datasetFile = h5py.File(pathF+nameDB+"_temp.hdf5", "r")
    return datasetFile, labelsDictionary


def getCaltechDBcsv(views, pathF, nameDB, NB_CLASS, LABELS_NAMES):
    datasetFile = h5py.File(pathF+nameDB+".hdf5", "w")
    labelsNamesFile = open(pathF+nameDB+'-ClassLabels-Description.csv')
    if len(LABELS_NAMES)!=NB_CLASS:
        nbLabelsAvailable = 0
        for l in labelsNamesFile:
            nbLabelsAvailable+=1
        LABELS_NAMES = [line.strip().split(";")[1] for lineIdx, line in enumerate(labelsNamesFile) if lineIdx in np.random.randint(nbLabelsAvailable, size=NB_CLASS)]
    fullLabels = np.genfromtxt(pathF + nameDB + '-ClassLabels.csv', delimiter=';').astype(int)
    labelsDictionary = dict((classIndice, labelName) for (classIndice, labelName) in
                            [(int(line.strip().split(";")[0]),line.strip().split(";")[1])for lineIndex, line in labelsNamesFile if line.strip().split(";")[0] in LABELS_NAMES])
    if len(set(fullLabels))>NB_CLASS:
        usedIndices = getPositions(labelsDictionary.keys(), fullLabels)
    else:
        usedIndices = range(len(fullLabels))
    for viewIndex, view in enumerate(views):
        viewFile = pathF + nameDB + "-" + view + '.csv'
        viewMatrix = np.array(np.genfromtxt(viewFile, delimiter=';'))[usedIndices, :]
        viewDset = datasetFile.create_dataset("View"+str(viewIndex), viewMatrix.shape, data=viewMatrix)
        viewDset.attrs["name"] = view

    labelsDset = datasetFile.create_dataset("Labels", fullLabels[usedIndices].shape, data=fullLabels[usedIndices])

    metaDataGrp = datasetFile.create_group("Metadata")
    metaDataGrp.attrs["nbView"] = len(views)
    metaDataGrp.attrs["nbClass"] = NB_CLASS
    metaDataGrp.attrs["datasetLength"] = len(fullLabels[usedIndices])
    datasetFile.close()
    datasetFile = h5py.File(pathF+nameDB+".hdf5", "r")
    return datasetFile, labelsDictionary


def getMultiOmicDBcsv(features, path, name, NB_CLASS, LABELS_NAMES):

    datasetFile = h5py.File(path+"MultiOmic.hdf5", "w")

    logging.debug("Start:\t Getting Methylation Data")
    methylData = np.genfromtxt(path+"matching_methyl.csv", delimiter=',')
    methylDset = datasetFile.create_dataset("View0", methylData.shape)
    methylDset[...] = methylData
    methylDset.attrs["name"] = "Methyl"
    methylDset.attrs["sparse"] = False
    logging.debug("Done:\t Getting Methylation Data")

    logging.debug("Start:\t Getting MiRNA Data")
    mirnaData = np.genfromtxt(path+"matching_mirna.csv", delimiter=',')
    mirnaDset = datasetFile.create_dataset("View1", mirnaData.shape)
    mirnaDset[...] = mirnaData
    mirnaDset.attrs["name"]="MiRNA_"
    mirnaDset.attrs["sparse"] = False
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
    rnaseqDset.attrs["sparse"] = False
    logging.debug("Done:\t Getting RNASeq Data")

    logging.debug("Start:\t Getting Clinical Data")
    clinical = np.genfromtxt(path+"clinicalMatrix.csv", delimiter=',')
    clinicalDset = datasetFile.create_dataset("View3", clinical.shape)
    clinicalDset[...] = clinical
    clinicalDset.attrs["name"] = "Clinic"
    clinicalDset.attrs["sparse"] = False
    logging.debug("Done:\t Getting Clinical Data")

    labelFile = open(path+'brca_labels_triple-negatif.csv')
    labels = np.array([int(line.strip().split(',')[1]) for line in labelFile])
    labelsDset = datasetFile.create_dataset("Labels", labels.shape)
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


def findClosestPowerOfTwo(k):
    power=1
    while k-power>0:
        power = 2*power
    if abs(k-power)<abs(k-power/2):
        return power
    else:
        return power/2


def getVector(matrix):
    argmax = [0,0]
    n = len(matrix)
    maxi = 0
    for i in range(n):
        for j in range(n):
            if j==i+1:
                value = (i+1)*(n-j)
                if value>maxi:
                    maxi= value
                    argmax = [i,j]
    i,j = argmax
    vector = np.zeros(n, dtype=bool)
    vector[:i+1]=np.ones(i+1, dtype=bool)
    matrixSup = [i+1, j+1]
    matrixInf = [i+1, j+1]
    return vector, matrixSup, matrixInf


def easyFactorize(targetMatrix, k, t=0):
    n = len(targetMatrix)
    if math.log(k+1, 2)%1==0.0:
        pass
    else:
        k = findClosestPowerOfTwo(k)-1
    if k==1:
        t=1
        return t, getVector(targetMatrix)[0]
    vector, matrixSup, matrixInf = getVector(targetMatrix)
    t, vectorSup = easyFactorize(targetMatrix[:matrixSup[0], :matrixSup[1]], (k-1)/2, t)
    t, vectorInf = easyFactorize(targetMatrix[matrixInf[0]:, matrixInf[0]:], (k-1)/2, t)
    factor = np.zeros((n,2*t+1), dtype=bool)
    factor[:matrixSup[0], :t] = vectorSup.reshape(factor[:matrixSup[0], :t].shape)
    factor[matrixInf[0]:, t:2*t] = vectorInf.reshape(factor[matrixInf[0]:, t:2*t].shape)
    factor[:, 2*t] = vector
    return 2*t+1, factor


def findParams(arrayLen, nbPatients, maxNbBins=5000, maxLenBin=300, minOverlapping=30, minNbBinsOverlapped=20, maxNbSolutions=30):
    results = []
    if arrayLen*arrayLen*10/100>minNbBinsOverlapped*nbPatients:
        for lenBin in range(arrayLen-1):
            if lenBin+1<maxLenBin:
                for overlapping in sorted(range(lenBin+1-1), reverse=True):
                    if overlapping+1>minOverlapping and overlapping>lenBin/minNbBinsOverlapped:
                        for nbBins in sorted(range(arrayLen-1), reverse=True):
                            if nbBins+1<maxNbBins:
                                if arrayLen == (nbBins+1-1)*(lenBin+1-overlapping+1)+lenBin+1:
                                    results.append({"nbBins":nbBins, "overlapping":overlapping, "lenBin":lenBin})
                                    if len(results)==maxNbSolutions:
                                        params = results[random.randrange(len(results))]
                                        return params


def findBins(nbBins, overlapping, lenBin):
    bins = []
    for binIndex in range(nbBins+1):
        bins.append([i+binIndex*(lenBin+1-overlapping+1) for i in range(lenBin+1)])
    return bins


def getBins(array, bins):
    binnedcoord = []
    for coordIndex, coord in enumerate(array):
        for binIndex, bin in enumerate(bins):
            if coordIndex in bin:
                binnedcoord.append(binIndex+coord*len(bins))
    return np.array(binnedcoord)


def makeSparseTotalMatrix(sortedRNASeq):
    nbPatients, nbGenes = sortedRNASeq.shape
    params = findParams(nbGenes, nbPatients)
    nbBins = params["nbBins"]
    overlapping = params["overlapping"]
    lenBin = params["lenBin"]
    bins = findBins(nbBins, overlapping, lenBin)
    sparseFull = sparse.csc_matrix((nbPatients, nbGenes*nbBins))
    for patientIndex, patient in enumerate(sortedRNASeq):
        print patientIndex
        binnedcoord = getBins(patient, bins)
        columIndices = binnedcoord
        rowIndices = np.zeros(len(binnedcoord), dtype=int)+patientIndex
        data = np.ones(len(binnedcoord), dtype=bool)
        sparseFull = sparseFull+sparse.csc_matrix((data, (rowIndices, columIndices)), shape=(nbPatients, nbGenes*nbBins))
    return sparseFull



def getModifiedMultiOmicDBcsv(features, path, name, NB_CLASS, LABELS_NAMES):

    datasetFile = h5py.File(path+"ModifiedMultiOmic.hdf5", "w")

    logging.debug("Start:\t Getting Methylation Data")
    methylData = np.genfromtxt(path+"matching_methyl.csv", delimiter=',')
    methylDset = datasetFile.create_dataset("View0", methylData.shape)
    methylDset[...] = methylData
    methylDset.attrs["name"] = "Methyl_"
    methylDset.attrs["sparse"] = False
    logging.debug("Done:\t Getting Methylation Data")

    logging.debug("Start:\t Getting MiRNA Data")
    mirnaData = np.genfromtxt(path+"matching_mirna.csv", delimiter=',')
    mirnaDset = datasetFile.create_dataset("View1", mirnaData.shape)
    mirnaDset[...] = mirnaData
    mirnaDset.attrs["name"]="MiRNA__"
    mirnaDset.attrs["sparse"]=False
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
    rnaseqDset.attrs["sparse"]=False
    logging.debug("Done:\t Getting RNASeq Data")

    logging.debug("Start:\t Getting Clinical Data")
    clinical = np.genfromtxt(path+"clinicalMatrix.csv", delimiter=',')
    clinicalDset = datasetFile.create_dataset("View3", clinical.shape)
    clinicalDset[...] = clinical
    clinicalDset.attrs["name"] = "Clinic_"
    clinicalDset.attrs["sparse"] = False
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
    mrnaseqDset.attrs["sparse"] = False
    logging.debug("Done:\t Getting Sorted RNASeq Data")

    logging.debug("Start:\t Getting Binarized RNASeq Data")
    k=100
    factorizedSupBaseMatrix = np.genfromtxt(path+"factorSup--n-"+str(len(modifiedRNASeq))+"--k-"+str(k)+".csv", delimiter=',')
    factorizedLeftBaseMatrix = np.genfromtxt(path+"factorLeft--n-73599--k-100.csv", delimiter=',')
    brnaseqDset = datasetFile.create_dataset("View5", (len(modifiedRNASeq), len(modifiedRNASeq)*k*2), dtype=bool)
    for patientIndex, patientSortedArray in enumerate(modifiedRNASeq):
        patientMatrix = np.zeros((len(modifiedRNASeq), k*2), dtype=bool)
        for lineIndex, geneIndex in enumerate(patientSortedArray):
            patientMatrix[geneIndex]= np.concatenate(factorizedLeftBaseMatrix[lineIndex], factorizedSupBaseMatrix[:, lineIndex])
        brnaseqDset[patientIndex] = patientMatrix.flatten()
    brnaseqDset.attrs["name"] = "bRNASeq"
    brnaseqDset.attrs["sparse"] = False
    logging.debug("Done:\t Getting Binarized RNASeq Data")

    logging.debug("Start:\t Getting Binned RNASeq Data")
    sparseBinnedRNASeq = makeSparseTotalMatrix(modifiedRNASeq)
    sparseBinnedRNASeqGrp = datasetFile.create_group("View6")
    dataDset = sparseBinnedRNASeqGrp.create_dataset("data", sparseBinnedRNASeq.data.shape, data=sparseBinnedRNASeq.data)
    indicesDset = sparseBinnedRNASeqGrp.create_dataset("indices", sparseBinnedRNASeq.indices.shape, data=sparseBinnedRNASeq.indices)
    indptrDset = sparseBinnedRNASeqGrp.create_dataset("indptr", sparseBinnedRNASeq.indptr.shape, data=sparseBinnedRNASeq.indptr)
    sparseBinnedRNASeqGrp.attrs["name"]="BRNASeq"
    sparseBinnedRNASeqGrp.attrs["sparse"]=True
    sparseBinnedRNASeqGrp.attrs["shape"]=sparseBinnedRNASeq.shape
    logging.debug("Done:\t Getting Binned RNASeq Data")

    labelFile = open(path+'brca_labels_triple-negatif.csv')
    labels = np.array([int(line.strip().split(',')[1]) for line in labelFile])
    labelsDset = datasetFile.create_dataset("Labels", labels.shape)
    labelsDset[...] = labels
    labelsDset.attrs["name"] = "Labels"

    metaDataGrp = datasetFile.create_group("Metadata")
    metaDataGrp.attrs["nbView"] = 5
    metaDataGrp.attrs["nbClass"] = 2
    metaDataGrp.attrs["datasetLength"] = len(labels)
    labelDictionary = {0:"No", 1:"Yes"}

    datasetFile.close()
    datasetFile = h5py.File(path+"ModifiedMultiOmic.hdf5", "r")

    return datasetFile, labelDictionary


def getModifiedMultiOmicDBhdf5(features, path, name, NB_CLASS, LABELS_NAMES):
    datasetFile = h5py.File(path+"ModifiedMultiOmic.hdf5", "r")
    labelDictionary = {0:"No", 1:"Yes"}
    return datasetFile, labelDictionary


def getMultiOmicDBhdf5(features, path, name, NB_CLASS, LABELS_NAMES):
    datasetFile = h5py.File(path+"MultiOmic.hdf5", "r")
    labelDictionary = {0:"No", 1:"Yes"}
    return datasetFile, labelDictionary


def copyHDF5(pathF, name, nbCores):
    datasetFile = h5py.File(pathF+name+".hdf5", "r")
    for coreIndex in range(nbCores):
        newDataSet = h5py.File(pathF+name+str(coreIndex)+".hdf5", "w")
        for dataset in datasetFile:
            datasetFile.copy("/"+dataset, newDataSet["/"])
        newDataSet.close()


def deleteHDF5(pathF, name, nbCores):
    for coreIndex in range(nbCores):
        os.remove(pathF+name+str(coreIndex)+".hdf5")

# def getOneViewFromDB(viewName, pathToDB, DBName):
#     view = np.genfromtxt(pathToDB + DBName +"-" + viewName, delimiter=';')
#     return view


# def getClassLabels(pathToDB, DBName):
#     labels = np.genfromtxt(pathToDB + DBName + "-" + "ClassLabels.csv", delimiter=';')
#     return labels


# def getDataset(pathToDB, viewNames, DBName):
#     dataset = []
#     for viewName in viewNames:
#         dataset.append(getOneViewFromDB(viewName, pathToDB, DBName))
#     return np.array(dataset)


# def getAwaLabels(nbLabels, pathToAwa):
#     labelsFile = open(pathToAwa + 'Animals_with_Attributes/classes.txt', 'U')
#     linesFile = [''.join(line.strip().split()).translate(None, digits) for line in labelsFile.readlines()]
#     return linesFile


# def getAwaDBcsv(views, pathToAwa, nameDB, nbLabels, LABELS_NAMES):
#     awaLabels = getAwaLabels(nbLabels, pathToAwa)
#     nbView = len(views)
#     nbMaxLabels = len(awaLabels)
#     if nbLabels == -1:
#         nbLabels = nbMaxLabels
#     nbNamesGiven = len(LABELS_NAMES)
#     if nbNamesGiven > nbLabels:
#         labelDictionary = {i:LABELS_NAMES[i] for i in np.arange(nbLabels)}
#     elif nbNamesGiven < nbLabels and nbLabels <= nbMaxLabels:
#         if LABELS_NAMES != ['']:
#             labelDictionary = {i:LABELS_NAMES[i] for i in np.arange(nbNamesGiven)}
#         else:
#             labelDictionary = {}
#             nbNamesGiven = 0
#         nbLabelsToAdd = nbLabels-nbNamesGiven
#         while nbLabelsToAdd > 0:
#             currentLabel = random.choice(awaLabels)
#             if currentLabel not in labelDictionary.values():
#                 labelDictionary[nbLabels-nbLabelsToAdd]=currentLabel
#                 nbLabelsToAdd -= 1
#             else:
#                 pass
#     else:
#         labelDictionary = {i: LABELS_NAMES[i] for i in np.arange(nbNamesGiven)}
#     viewDictionary = {i: views[i] for i in np.arange(nbView)}
#     rawData = []
#     labels = []
#     nbExample = 0
#     for view in np.arange(nbView):
#         viewData = []
#         for labelIndex in np.arange(nbLabels):
#             pathToExamples = pathToAwa + 'Animals_with_Attributes/Features/' + viewDictionary[view] + '/' + \
#                              labelDictionary[labelIndex] + '/'
#             examples = os.listdir(pathToExamples)
#             if view == 0:
#                 nbExample += len(examples)
#             for example in examples:
#                 if viewDictionary[view]=='decaf':
#                     exampleFile = open(pathToExamples + example)
#                     viewData.append([float(line.strip()) for line in exampleFile])
#                 else:
#                     exampleFile = open(pathToExamples + example)
#                     viewData.append([[float(coordinate) for coordinate in raw.split()] for raw in exampleFile][0])
#                 if view == 0:
#                     labels.append(labelIndex)
#
#         rawData.append(np.array(viewData))
#     data = rawData
#     DATASET_LENGTH = len(labels)
#     return data, labels, labelDictionary, DATASET_LENGTH
#
#
# def getDbfromCSV(path):
#     files = os.listdir(path)
#     DATA = np.zeros((3,40,2))
#     for file in files:
#         if file[-9:]=='moins.csv' and file[:7]=='sample1':
#             X = open(path+file)
#             for x, i in zip(X, range(20)):
#                 DATA[0, i] = np.array([float(coord) for coord in x.strip().split('\t')])
#         if file[-9:]=='moins.csv' and file[:7]=='sample2':
#             X = open(path+file)
#             for x, i in zip(X, range(20)):
#                 DATA[1, i] = np.array([float(coord) for coord in x.strip().split('\t')])
#         if file[-9:]=='moins.csv' and file[:7]=='sample3':
#             X = open(path+file)
#             for x, i in zip(X, range(20)):
#                 DATA[2, i] = np.array([float(coord) for coord in x.strip().split('\t')])
#
#     for file in files:
#         if file[-8:]=='plus.csv' and file[:7]=='sample1':
#             X = open(path+file)
#             for x, i in zip(X, range(20)):
#                 DATA[0, i+20] = np.array([float(coord) for coord in x.strip().split('\t')])
#         if file[-8:]=='plus.csv' and file[:7]=='sample2':
#             X = open(path+file)
#             for x, i in zip(X, range(20)):
#                 DATA[1, i+20] = np.array([float(coord) for coord in x.strip().split('\t')])
#         if file[-8:]=='plus.csv' and file[:7]=='sample3':
#             X = open(path+file)
#             for x, i in zip(X, range(20)):
#                 DATA[2, i+20] = np.array([float(coord) for coord in x.strip().split('\t')])
#     LABELS = np.zeros(40)
#     LABELS[:20]=LABELS[:20]+1
#     return DATA, LABELS

# def makeArrayFromTriangular(pseudoRNASeqMatrix):
#     matrixShape = len(pseudoRNASeqMatrix[0,:])
#     exampleArray = np.array(((matrixShape-1)*matrixShape)/2)
#     arrayIndex = 0
#     for i in range(matrixShape-1):
#         for j in range(i+1, matrixShape):
#             exampleArray[arrayIndex]=pseudoRNASeqMatrix[i,j]
#             arrayIndex += 1
#     return exampleArray


# def getPseudoRNASeq(dataset):
#     nbGenes = len(dataset["/View2/matrix"][0, :])
#     pseudoRNASeq = np.zeros((dataset["/datasetlength"][...], ((nbGenes - 1) * nbGenes) / 2), dtype=bool_)
#     for exampleIndex in xrange(dataset["/datasetlength"][...]):
#         arrayIndex = 0
#         for i in xrange(nbGenes):
#             for j in xrange(nbGenes):
#                 if i > j:
#                     pseudoRNASeq[exampleIndex, arrayIndex] = dataset["/View2/matrix"][exampleIndex, j] < dataset["/View2/matrix"][exampleIndex, i]
#                     arrayIndex += 1
#     dataset["/View4/matrix"] = pseudoRNASeq
#     dataset["/View4/name"] = "pseudoRNASeq"
#     return dataset


# def allSame(array):
#     value = array[0]
#     areAllSame = True
#     for i in array:
#         if i != value:
#             areAllSame = False
#     return areAllSame

