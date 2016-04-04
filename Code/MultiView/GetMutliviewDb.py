import numpy as np

def getOneViewFromDB(viewName, pathToDB):
	view = np.genfromtxt(pathToDB + viewName, delimiter=';')
	return view

def getClassLabels(pathToDB):
	labels = np.genfromtxt(pathToDB + "ClassLabels.csv", delimiter=';')
	return labels 

def getDataset(pathToDB, viewNames):
	dataset = []
	for viewName in viewNames: 
		dataset.append(getOneViewFromDB(viewName, pathtoDB))
	return np.array(dataset)
