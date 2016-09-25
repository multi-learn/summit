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


def getPlausibleDBhdf5(features, pathF, name , NB_CLASS, LABELS_NAME, nbView=4, nbClass=2, datasetLength=100):
    nbFeatures = 300
    datasetFile = h5py.File(pathF+"Plausible.hdf5", "w")
    for viewIndex in range(nbView):
        if viewIndex== 0 :
            viewData = np.array([np.zeros(nbFeatures) for i in range(datasetLength/2)]+[np.ones(nbFeatures) for i in range(datasetLength/2)]).astype(np.uint8)
            viewDset = datasetFile.create_dataset("View"+str(viewIndex), viewData.shape)
            viewDset.attrs["name"] = "View"+str(viewIndex)
            viewDset.attrs["sparse"] = False
            viewDset.attrs["binary"] = True
        else:
            viewData = np.array([np.random.normal(float((viewIndex+1)*10), 0.42, nbFeatures) for i in range(datasetLength/2)]+[np.random.normal(-float((viewIndex+1)*10),0.42,nbFeatures) for j in range(datasetLength/2)])
            viewDset = datasetFile.create_dataset("View"+str(viewIndex), viewData.shape)
            viewDset.attrs["name"] = "View"+str(viewIndex)
            viewDset.attrs["sparse"] = False
            viewDset.attrs["binary"] = False
    CLASS_LABELS = np.array([0 for i in range(datasetLength/2+5)]+[1 for i in range(datasetLength/2-5)])
    labelsDset = datasetFile.create_dataset("Labels", CLASS_LABELS.shape)
    labelsDset[...] = CLASS_LABELS
    labelsDset.attrs["name"] = "Labels"
    metaDataGrp = datasetFile.create_group("Metadata")
    metaDataGrp.attrs["nbView"] = nbView
    metaDataGrp.attrs["nbClass"] = 2
    metaDataGrp.attrs["datasetLength"] = len(CLASS_LABELS)
    datasetFile.close()
    datasetFile = h5py.File(pathF+"Plausible.hdf5", "r")
    LABELS_DICTIONARY = {0:"No", 1:"Yes"}
    return datasetFile, LABELS_DICTIONARY


def getFakeDBhdf5(features, pathF, name , NB_CLASS, LABELS_NAME):
    NB_VIEW = 4
    DATASET_LENGTH = 300
    NB_CLASS = 2
    VIEW_DIMENSIONS = np.random.random_integers(5, 20, NB_VIEW)

    DATA = dict((indx,
                        np.array([
                                     np.random.normal(0.0, 2, viewDimension)
                                     for i in np.arange(DATASET_LENGTH)]))
                        for indx, viewDimension in enumerate(VIEW_DIMENSIONS))

    CLASS_LABELS = np.random.random_integers(0, NB_CLASS-1, DATASET_LENGTH)
    datasetFile = h5py.File(pathF+"Fake.hdf5", "w")
    for index, viewData in enumerate(DATA.values()):
        if index==0:
            viewData = np.random.randint(0, 1, (DATASET_LENGTH,300)).astype(np.uint8)#np.zeros(viewData.shape, dtype=bool)+np.ones((viewData.shape[0], viewData.shape[1]/2), dtype=bool)
            viewDset = datasetFile.create_dataset("View"+str(index), viewData.shape)
            viewDset[...] = viewData
            viewDset.attrs["name"] = "View"+str(index)
            viewDset.attrs["sparse"] = False
        elif index == 1:
            viewData = sparse.csr_matrix(viewData)
            viewGrp = datasetFile.create_group("View"+str(index))
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
    LABELS_DICTIONARY = {0:"No", 1:"Yes"}
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


def getVector(nbGenes):
    argmax = [0,0]
    maxi = 0
    for i in range(nbGenes):
        for j in range(nbGenes):
            if j==i+1:
                value = (i+1)*(nbGenes-j)
                if value>maxi:
                    maxi= value
                    argmax = [i,j]
    i,j = argmax
    vectorLeft = np.zeros(nbGenes, dtype=bool)
    vectorLeft[:i+1] = np.ones(i+1, dtype=bool)
    vectorSup = np.zeros(nbGenes, dtype=bool)
    vectorSup[j:] = np.ones(nbGenes-j, dtype=bool)
    matrixSup = j
    matrixInf = nbGenes-j
    return vectorLeft, matrixSup, matrixInf


def findClosestPowerOfTwo(factorizationParam):
    power=1
    while factorizationParam-power>0:
        power = 2*power
    if abs(factorizationParam-power)<abs(factorizationParam-power/2):
        return power
    else:
        return power/2


def easyFactorize(nbGenes, factorizationParam, t=0):
    if math.log(factorizationParam+1, 2)%1==0.0:
        pass
    else:
        factorizationParam = findClosestPowerOfTwo(factorizationParam) - 1

    if nbGenes==2:
        return 1, np.array([True, False])

    if nbGenes==3:
        return 1, np.array([True, True,  False])

    if factorizationParam==1:
        t=1
        return t, getVector(nbGenes)[0]

    vectorLeft, matrixSup, matrixInf = getVector(nbGenes)

    t_, vectorLeftSup = easyFactorize(matrixSup, (factorizationParam - 1) / 2, t=t)
    t__, vectorLeftInf = easyFactorize(matrixInf, (factorizationParam - 1) / 2, t=t)

    factorLeft = np.zeros((nbGenes,t_+t__+1), dtype=bool)

    factorLeft[:matrixSup, :t_] = vectorLeftSup.reshape(factorLeft[:matrixSup, :t_].shape)
    if nbGenes%2==1:
        factorLeft[matrixInf-1:, t_:t__+t_] = vectorLeftInf.reshape(factorLeft[matrixInf-1:, t_:t__+t_].shape)
    else:
        factorLeft[matrixInf:, t_:t__+t_] = vectorLeftInf.reshape(factorLeft[matrixInf:, t_:t__+t_].shape)
    factorLeft[:, t__+t_] = vectorLeft

    # factorSup = np.zeros((t_+t__+1, nbGenes), dtype=bool)
    #
    # factorSup[:t_, :matrixSup] = vectorSupLeft.reshape(factorSup[:t_, :matrixSup].shape)
    # if nbGenes%2==1:
    #     factorSup[t_:t__+t_, matrixInf-1:] = vectorSupRight.reshape(factorSup[t_:t__+t_, matrixInf-1:].shape)
    # else:
    #     factorSup[t_:t__+t_, matrixInf:] = vectorSupRight.reshape(factorSup[t_:t__+t_, matrixInf:].shape)
    # factorSup[t__+t_, :] = vectorSup
    return t__+t_+1, factorLeft#, factorSup


def getBaseMatrices(nbGenes, factorizationParam, path):
    t, factorLeft = easyFactorize(nbGenes, factorizationParam)
    np.savetxt(path+"factorLeft--n-"+str(nbGenes)+"--k-"+str(factorizationParam)+".csv", factorLeft, delimiter=",")
    return factorLeft


def findParams(arrayLen, nbPatients, maxNbBins=2000, minNbBins = 10, maxLenBin=70000, minOverlapping=1, minNbBinsOverlapped=0, maxNbSolutions=30):
    results = []
    if arrayLen*arrayLen*10/100>minNbBinsOverlapped*nbPatients:
        for lenBin in range(arrayLen-1):
            lenBin = lenBin+1
            if lenBin<maxLenBin and minNbBins*lenBin<arrayLen:
                print lenBin
                print results
                for overlapping in sorted(range(lenBin-1), reverse=True):
                    overlapping = overlapping+1
                    if overlapping>minOverlapping and lenBin%(lenBin-overlapping)==0:
                        for nbBins in sorted(range(arrayLen-1), reverse=True):
                            nbBins = nbBins+1
                            if nbBins<maxNbBins:
                                if arrayLen == (nbBins-1)*(lenBin-overlapping)+lenBin:
                                    results.append({"nbBins":nbBins, "overlapping":overlapping, "lenBin":lenBin})
                                    if len(results)==maxNbSolutions:
                                        params = results[random.randrange(len(results))]
                                        return params


def findBins(nbBins=142, overlapping=493, lenBin=986):
    bins = []
    for binIndex in range(nbBins+1):
        bins.append([i+binIndex*(lenBin+1-overlapping+1) for i in range(lenBin+1)])
    return bins


def getBins(array, bins, lenBin, overlapping):
    binnedcoord = []
    for coordIndex, coord in enumerate(array):
        nbBinsFull = 0
        for binIndex, bin in enumerate(bins):
           if coordIndex in bin:
                binnedcoord.append(binIndex+(coord*len(bins)))

    return np.array(binnedcoord)


def makeSortedBinsMatrix(nbBins, lenBins, overlapping, arrayLen, path):
    sortedBinsMatrix = np.zeros((arrayLen, nbBins), dtype=np.uint8)
    step = lenBins-overlapping
    for binIndex in range(nbBins):
        sortedBinsMatrix[step*binIndex:lenBins+(step*binIndex), binIndex] = np.ones(lenBins, dtype=np.uint8)
    np.savetxt(path+"sortedBinsMatrix--t-"+str(lenBins)+"--n-"+str(nbBins)+"--c-"+str(overlapping)+".csv", sortedBinsMatrix, delimiter=",")
    return sortedBinsMatrix

def makeSparseTotalMatrix(sortedRNASeq):
    nbPatients, nbGenes = sortedRNASeq.shape
    params = findParams(nbGenes, nbPatients)
    nbBins = params["nbBins"]
    overlapping = params["overlapping"]
    lenBin = params["lenBin"]
    bins = findBins(nbBins, overlapping, lenBin)
    sparseFull = sparse.csc_matrix((nbPatients, nbGenes*nbBins))
    for patientIndex, patient in enumerate(sortedRNASeq):
        columnIndices = getBins(patient, bins, lenBin, overlapping)
        rowIndices = np.zeros(len(columnIndices), dtype=int)+patientIndex
        data = np.ones(len(columnIndices), dtype=bool)
        sparseFull = sparseFull+sparse.csc_matrix((data, (rowIndices, columnIndices)), shape=(nbPatients, nbGenes*nbBins))
    return sparseFull


def getAdjacenceMatrix(RNASeqRanking, sotredRNASeq, k=2):
    k=int(k)/2*2
    indices = np.zeros((RNASeqRanking.shape[0]*k*RNASeqRanking.shape[1]), dtype=int)
    data = np.ones((RNASeqRanking.shape[0]*k*RNASeqRanking.shape[1]), dtype=bool)
    indptr = np.zeros(RNASeqRanking.shape[0]+1, dtype=int)
    nbGenes = RNASeqRanking.shape[1]
    pointer = 0
    for patientIndex in range(RNASeqRanking.shape[0]):
        print patientIndex
        for i in range(nbGenes):
            for j in range(k/2):
                try:
                    indices[pointer]=RNASeqRanking[patientIndex, (sotredRNASeq[patientIndex, i]-(j+1))]+i*nbGenes
                    pointer+=1
                except:
                    pass
                try:
                    indices[pointer]=RNASeqRanking[patientIndex, (sotredRNASeq[patientIndex, i]+(j+1))]+i*nbGenes
                    pointer+=1
                except:
                    pass
                    # elif i<=k:
                    # 	indices.append(patient[1]+patient[i]*nbGenes)
                    # 	data.append(True)
                    # elif i==nbGenes-1:
                    # 	indices.append(patient[i-1]+patient[i]*nbGenes)
                    # 	data.append(True)
        indptr[patientIndex+1] = pointer

    mat = sparse.csr_matrix((data, indices, indptr), shape=(RNASeqRanking.shape[0], RNASeqRanking.shape[1]*RNASeqRanking.shape[1]), dtype=bool)
    return mat


def getModifiedMultiOmicDBcsv(features, path, name, NB_CLASS, LABELS_NAMES):

    datasetFile = h5py.File(path+"ModifiedMultiOmicb.hdf5", "w")

    logging.debug("Start:\t Getting Methylation Data")
    methylData = np.genfromtxt(path+"matching_methyl.csv", delimiter=',')
    methylDset = datasetFile.create_dataset("View0", methylData.shape)
    methylDset[...] = methylData
    methylDset.attrs["name"] = "Methyl_"
    methylDset.attrs["sparse"] = False
    methylDset.attrs["binary"] = False
    logging.debug("Done:\t Getting Methylation Data")

    logging.debug("Start:\t Getting Sorted Methyl Data")
    Methyl = datasetFile["View0"][...]
    sortedMethylGeneIndices = np.zeros(datasetFile.get("View0").shape, dtype=int)
    MethylRanking = np.zeros(datasetFile.get("View0").shape, dtype=int)
    for exampleIndex, exampleArray in enumerate(Methyl):
        sortedMethylDictionary = dict((index, value) for index, value in enumerate(exampleArray))
        sortedMethylIndicesDict = sorted(sortedMethylDictionary.items(), key=operator.itemgetter(1))
        sortedMethylIndicesArray = np.array([index for (index, value) in sortedMethylIndicesDict], dtype=int)
        sortedMethylGeneIndices[exampleIndex] = sortedMethylIndicesArray
        for geneIndex in range(Methyl.shape[1]):
            MethylRanking[exampleIndex, sortedMethylIndicesArray[geneIndex]] = geneIndex
    mMethylDset = datasetFile.create_dataset("View10", sortedMethylGeneIndices.shape, data=sortedMethylGeneIndices)
    mMethylDset.attrs["name"] = "SMethyl"
    mMethylDset.attrs["sparse"] = False
    mMethylDset.attrs["binary"] = False
    logging.debug("Done:\t Getting Sorted Methyl Data")


    logging.debug("Start:\t Getting Binarized Methyl Data")
    k=findClosestPowerOfTwo(58)-1
    try:
        factorizedLeftBaseMatrix = np.genfromtxt(path+"factorLeft--n-"+str(datasetFile.get("View0").shape[1])+"--k-"+str(k)+".csv", delimiter=',')
    except:
        factorizedLeftBaseMatrix = getBaseMatrices(methylData.shape[1], k, path)
    bMethylDset = datasetFile.create_dataset("View11", (sortedMethylGeneIndices.shape[0], sortedMethylGeneIndices.shape[1]*k), dtype=np.uint8)
    for patientIndex, patientSortedArray in enumerate(sortedMethylGeneIndices):
        patientMatrix = np.zeros((sortedMethylGeneIndices.shape[1], k), dtype=np.uint8)
        for lineIndex, geneIndex in enumerate(patientSortedArray):
            patientMatrix[geneIndex]= factorizedLeftBaseMatrix[lineIndex,:]
        bMethylDset[patientIndex] = patientMatrix.flatten()
    bMethylDset.attrs["name"] = "BMethyl"
    bMethylDset.attrs["sparse"] = False
    bMethylDset.attrs["binary"] = True
    logging.debug("Done:\t Getting Binarized Methyl Data")

    logging.debug("Start:\t Getting Binned Methyl Data")
    lenBins = 2095
    nbBins = 58
    overlapping = 1676
    try:
        sortedBinsMatrix = np.genfromtxt(path+"sortedBinsMatrix--t-"+str(lenBins)+"--n-"+str(nbBins)+"--c-"+str(overlapping)+".csv", delimiter=",")
    except:
        sortedBinsMatrix = makeSortedBinsMatrix(nbBins, lenBins, overlapping, datasetFile.get("View0").shape[1], path)
    binnedMethyl = datasetFile.create_dataset("View12", (sortedMethylGeneIndices.shape[0], sortedMethylGeneIndices.shape[1]*nbBins), dtype=np.uint8)
    for patientIndex, patientSortedArray in enumerate(sortedMethylGeneIndices):
        patientMatrix = np.zeros((sortedMethylGeneIndices.shape[1], nbBins), dtype=np.uint8)
        for lineIndex, geneIndex in enumerate(patientSortedArray):
            patientMatrix[geneIndex]= sortedBinsMatrix[lineIndex,:]
        binnedMethyl[patientIndex] = patientMatrix.flatten()
    binnedMethyl.attrs["name"] = "bMethyl"
    binnedMethyl.attrs["sparse"] = False
    binnedMethyl.attrs["binary"] = True
    logging.debug("Done:\t Getting Binned Methyl Data")

    logging.debug("Start:\t Getting MiRNA Data")
    mirnaData = np.genfromtxt(path+"matching_mirna.csv", delimiter=',')
    mirnaDset = datasetFile.create_dataset("View1", mirnaData.shape)
    mirnaDset[...] = mirnaData
    mirnaDset.attrs["name"]="MiRNA__"
    mirnaDset.attrs["sparse"]=False
    mirnaDset.attrs["binary"] = False
    logging.debug("Done:\t Getting MiRNA Data")

    logging.debug("Start:\t Getting Sorted MiRNA Data")
    MiRNA = datasetFile["View1"][...]
    sortedMiRNAGeneIndices = np.zeros(datasetFile.get("View1").shape, dtype=int)
    MiRNARanking = np.zeros(datasetFile.get("View1").shape, dtype=int)
    for exampleIndex, exampleArray in enumerate(MiRNA):
        sortedMiRNADictionary = dict((index, value) for index, value in enumerate(exampleArray))
        sortedMiRNAIndicesDict = sorted(sortedMiRNADictionary.items(), key=operator.itemgetter(1))
        sortedMiRNAIndicesArray = np.array([index for (index, value) in sortedMiRNAIndicesDict], dtype=int)
        sortedMiRNAGeneIndices[exampleIndex] = sortedMiRNAIndicesArray
        for geneIndex in range(MiRNA.shape[1]):
            MiRNARanking[exampleIndex, sortedMiRNAIndicesArray[geneIndex]] = geneIndex
    mmirnaDset = datasetFile.create_dataset("View7", sortedMiRNAGeneIndices.shape, data=sortedMiRNAGeneIndices)
    mmirnaDset.attrs["name"] = "SMiRNA_"
    mmirnaDset.attrs["sparse"] = False
    mmirnaDset.attrs["binary"] = False
    logging.debug("Done:\t Getting Sorted MiRNA Data")


    logging.debug("Start:\t Getting Binarized MiRNA Data")
    k=findClosestPowerOfTwo(517)-1
    try:
        factorizedLeftBaseMatrix = np.genfromtxt(path+"factorLeft--n-"+str(datasetFile.get("View1").shape[1])+"--k-"+str(k)+".csv", delimiter=',')
    except:
        factorizedLeftBaseMatrix = getBaseMatrices(mirnaData.shape[1], k, path)
    bmirnaDset = datasetFile.create_dataset("View8", (sortedMiRNAGeneIndices.shape[0], sortedMiRNAGeneIndices.shape[1]*k), dtype=np.uint8)
    for patientIndex, patientSortedArray in enumerate(sortedMiRNAGeneIndices):
        patientMatrix = np.zeros((sortedMiRNAGeneIndices.shape[1], k), dtype=np.uint8)
        for lineIndex, geneIndex in enumerate(patientSortedArray):
            patientMatrix[geneIndex]= factorizedLeftBaseMatrix[lineIndex,:]
        bmirnaDset[patientIndex] = patientMatrix.flatten()
    bmirnaDset.attrs["name"] = "BMiRNA_"
    bmirnaDset.attrs["sparse"] = False
    bmirnaDset.attrs["binary"] = True
    logging.debug("Done:\t Getting Binarized MiRNA Data")

    logging.debug("Start:\t Getting Binned MiRNA Data")
    lenBins = 14
    nbBins = 517
    overlapping = 12
    try:
        sortedBinsMatrix = np.genfromtxt(path+"sortedBinsMatrix--t-"+str(lenBins)+"--n-"+str(nbBins)+"--c-"+str(overlapping)+".csv", delimiter=",")
    except:
        sortedBinsMatrix = makeSortedBinsMatrix(nbBins, lenBins, overlapping, datasetFile.get("View1").shape[1], path)
    binnedMiRNA = datasetFile.create_dataset("View9", (sortedMiRNAGeneIndices.shape[0], sortedMiRNAGeneIndices.shape[1]*nbBins), dtype=np.uint8)
    for patientIndex, patientSortedArray in enumerate(sortedMiRNAGeneIndices):
        patientMatrix = np.zeros((sortedMiRNAGeneIndices.shape[1], nbBins), dtype=np.uint8)
        for lineIndex, geneIndex in enumerate(patientSortedArray):
            patientMatrix[geneIndex]= sortedBinsMatrix[lineIndex,:]
        binnedMiRNA[patientIndex] = patientMatrix.flatten()
    binnedMiRNA.attrs["name"] = "bMiRNA_"
    binnedMiRNA.attrs["sparse"] = False
    binnedMiRNA.attrs["binary"] = True
    logging.debug("Done:\t Getting Binned MiRNA Data")

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
    rnaseqDset.attrs["binary"] = False
    logging.debug("Done:\t Getting RNASeq Data")

    logging.debug("Start:\t Getting Sorted RNASeq Data")
    RNASeq = datasetFile["View2"][...]
    sortedRNASeqGeneIndices = np.zeros(datasetFile.get("View2").shape, dtype=int)
    RNASeqRanking = np.zeros(datasetFile.get("View2").shape, dtype=int)
    for exampleIndex, exampleArray in enumerate(RNASeq):
        sortedRNASeqDictionary = dict((index, value) for index, value in enumerate(exampleArray))
        sortedRNASeqIndicesDict = sorted(sortedRNASeqDictionary.items(), key=operator.itemgetter(1))
        sortedRNASeqIndicesArray = np.array([index for (index, value) in sortedRNASeqIndicesDict], dtype=int)
        sortedRNASeqGeneIndices[exampleIndex] = sortedRNASeqIndicesArray
        for geneIndex in range(RNASeq.shape[1]):
            RNASeqRanking[exampleIndex, sortedRNASeqIndicesArray[geneIndex]] = geneIndex
    mrnaseqDset = datasetFile.create_dataset("View4", sortedRNASeqGeneIndices.shape, data=sortedRNASeqGeneIndices)
    mrnaseqDset.attrs["name"] = "SRNASeq"
    mrnaseqDset.attrs["sparse"] = False
    mrnaseqDset.attrs["binary"] = False
    logging.debug("Done:\t Getting Sorted RNASeq Data")

    logging.debug("Start:\t Getting Binarized RNASeq Data")
    k=findClosestPowerOfTwo(100)-1
    try:
        factorizedLeftBaseMatrix = np.genfromtxt(path+"factorLeft--n-"+str(datasetFile.get("View2").shape[1])+"--k-"+str(100)+".csv", delimiter=',')
    except:
        factorizedLeftBaseMatrix = getBaseMatrices(rnaseqData.shape[1], k, path)
    brnaseqDset = datasetFile.create_dataset("View5", (sortedRNASeqGeneIndices.shape[0], sortedRNASeqGeneIndices.shape[1]*k), dtype=np.uint8)
    for patientIndex, patientSortedArray in enumerate(sortedRNASeqGeneIndices):
        patientMatrix = np.zeros((sortedRNASeqGeneIndices.shape[1], k), dtype=np.uint8)
        for lineIndex, geneIndex in enumerate(patientSortedArray):
            patientMatrix[geneIndex]= factorizedLeftBaseMatrix[lineIndex,:]
        brnaseqDset[patientIndex] = patientMatrix.flatten()
    brnaseqDset.attrs["name"] = "BRNASeq"
    brnaseqDset.attrs["sparse"] = False
    brnaseqDset.attrs["binary"] = True
    logging.debug("Done:\t Getting Binarized RNASeq Data")

    logging.debug("Start:\t Getting Binned RNASeq Data")
    lenBins = 986
    nbBins = 142
    overlapping = 493
    try:
        sortedBinsMatrix = np.genfromtxt(path+"sortedBinsMatrix--t-"+str(lenBins)+"--n-"+str(nbBins)+"--c-"+str(overlapping)+".csv", delimiter=",")
    except:
        sortedBinsMatrix = makeSortedBinsMatrix(nbBins, lenBins, overlapping, datasetFile.get("View2").shape[1], path)
    binnedRNASeq = datasetFile.create_dataset("View6", (sortedRNASeqGeneIndices.shape[0], sortedRNASeqGeneIndices.shape[1]*nbBins), dtype=np.uint8)
    for patientIndex, patientSortedArray in enumerate(sortedRNASeqGeneIndices):
        patientMatrix = np.zeros((sortedRNASeqGeneIndices.shape[1], nbBins), dtype=np.uint8)
        for lineIndex, geneIndex in enumerate(patientSortedArray):
            patientMatrix[geneIndex]= sortedBinsMatrix[lineIndex,:]
        binnedRNASeq[patientIndex] = patientMatrix.flatten()
    binnedRNASeq.attrs["name"] = "bRNASeq"
    binnedRNASeq.attrs["sparse"] = False
    binnedRNASeq.attrs["binary"] = True
    logging.debug("Done:\t Getting Binned RNASeq Data")

    logging.debug("Start:\t Getting Clinical Data")
    clinical = np.genfromtxt(path+"clinicalMatrix.csv", delimiter=',')
    clinicalDset = datasetFile.create_dataset("View3", clinical.shape)
    clinicalDset[...] = clinical
    clinicalDset.attrs["name"] = "Clinic_"
    clinicalDset.attrs["sparse"] = False
    clinicalDset.attrs["binary"] = False
    logging.debug("Done:\t Getting Clinical Data")

    logging.debug("Start:\t Getting Binarized Clinical Data")
    binarized_clinical = np.zeros((347,1951), dtype=np.uint8)
    nb_already_done = 0
    for feqtureIndex, feature in enumerate(np.transpose(clinical)):
        featureSet = set(feature)
        featureDict = dict((val,valIndex) for valIndex, val in enumerate(list(featureSet)))
        for valueIndex, value in enumerate(feature):
            binarized_clinical[valueIndex, featureDict[value]+nb_already_done] = 1
        nb_already_done+= len(featureSet)
    bClinicalDset = datasetFile.create_dataset("View13", binarized_clinical.shape, dtype=np.uint8, data=binarized_clinical)
    bClinicalDset.attrs["name"] = "bClinic"
    bClinicalDset.attrs["sparse"] = False
    bClinicalDset.attrs["binary"] = True
    logging.debug("Done:\t Getting Binarized Clinical Data")

    # logging.debug("Start:\t Getting Adjacence RNASeq Data")
    # sparseAdjRNASeq = getAdjacenceMatrix(RNASeqRanking, sortedRNASeqGeneIndices, k=findClosestPowerOfTwo(10)-1)
    # sparseAdjRNASeqGrp = datasetFile.create_group("View6")
    # dataDset = sparseAdjRNASeqGrp.create_dataset("data", sparseAdjRNASeq.data.shape, data=sparseAdjRNASeq.data)
    # indicesDset = sparseAdjRNASeqGrp.create_dataset("indices", sparseAdjRNASeq.indices.shape, data=sparseAdjRNASeq.indices)
    # indptrDset = sparseAdjRNASeqGrp.create_dataset("indptr", sparseAdjRNASeq.indptr.shape, data=sparseAdjRNASeq.indptr)
    # sparseAdjRNASeqGrp.attrs["name"]="ARNASeq"
    # sparseAdjRNASeqGrp.attrs["sparse"]=True
    # sparseAdjRNASeqGrp.attrs["shape"]=sparseAdjRNASeq.shape
    # logging.debug("Done:\t Getting Adjacence RNASeq Data")

    labelFile = open(path+'brca_labels_triple-negatif.csv')
    labels = np.array([int(line.strip().split(',')[1]) for line in labelFile])
    labelsDset = datasetFile.create_dataset("Labels", labels.shape)
    labelsDset[...] = labels
    labelsDset.attrs["name"] = "Labels"

    metaDataGrp = datasetFile.create_group("Metadata")
    metaDataGrp.attrs["nbView"] = 14
    metaDataGrp.attrs["nbClass"] = 2
    metaDataGrp.attrs["datasetLength"] = len(labels)
    labelDictionary = {0:"No", 1:"Yes"}

    datasetFile.close()
    datasetFile = h5py.File(path+"ModifiedMultiOmicb.hdf5", "r")

    return datasetFile, labelDictionary


def getModifiedMultiOmicDBhdf5(features, path, name, NB_CLASS, LABELS_NAMES):
    datasetFile = h5py.File(path+"ModifiedMultiOmicb.hdf5", "r")
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

def datasetsAlreadyExist(pathF, name, nbCores):
    allDatasetExist = True
    for coreIndex in range(nbCores):
        import os.path
        allDatasetExist *= os.path.isfile(pathF+name+str(coreIndex)+".hdf5")
    return allDatasetExist


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

