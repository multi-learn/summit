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