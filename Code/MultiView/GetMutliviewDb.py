import numpy as np


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