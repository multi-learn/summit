import numpy as np
import math
import itertools
import os

from ..utils.Multiclass import isBiclass, genMulticlassMonoviewDecision


def getClassifiersDecisions(allClassifersNames, viewsIndices, resultsMonoview):
    """
    This function gets the monoview classifiers decisions from resultsMonoview.
    The classifiersDecisions variable is ordered as :
    classifiersDecisions[viewIndex, classifierIndex, foldIndex, exampleIndex]
    And the classifiersNames variable is ordered as :
    classifiersNames[viewIndex][classifierIndex]
    """
    nbViews = len(viewsIndices)
    nbClassifiers = len(allClassifersNames)
    nbFolds = len(resultsMonoview[0][1][6])
    foldsLen = len(resultsMonoview[0][1][6][0])
    classifiersNames = [[] for _ in viewsIndices]
    classifiersDecisions = np.zeros((nbViews, nbClassifiers, nbFolds, foldsLen))

    for resultMonoview in resultsMonoview:
        if resultMonoview[1][0] in classifiersNames[viewsIndices.index(resultMonoview[0])]:
            pass
        else:
            classifiersNames[viewsIndices.index(resultMonoview[0])].append(resultMonoview[1][0])
        classifierIndex = classifiersNames[viewsIndices.index(resultMonoview[0])].index(resultMonoview[1][0])
        classifiersDecisions[viewsIndices.index(resultMonoview[0]), classifierIndex] = resultMonoview[1][6]
    return classifiersDecisions, classifiersNames


def couple_div_measure(allClassifersNames, viewsIndices, resultsMonoview, measurement, foldsGroudTruth):
    """
    This function is used to get the max of a couple diversity measurement,passed as an argument
    It generates all possible combinations and all the couples to estimate the diversity on a combination
    The best combination is the one that maximize the measurement.
    """

    classifiersDecisions, classifiersNames = getClassifiersDecisions(allClassifersNames,
                                                                                     viewsIndices,
                                                                                     resultsMonoview)

    nbViews = len(viewsIndices)
    nbClassifiers = len(allClassifersNames)
    combinations = itertools.combinations_with_replacement(range(nbClassifiers), nbViews)
    nbCombinations = int(math.factorial(nbClassifiers+nbViews-1) / math.factorial(nbViews) / math.factorial(nbClassifiers-1))
    div_measure = np.zeros(nbCombinations)
    combis = np.zeros((nbCombinations, nbViews), dtype=int)

    for combinationsIndex, combination in enumerate(combinations):
        combis[combinationsIndex] = combination
        combiWithView = [(viewIndex,combiIndex) for viewIndex, combiIndex in enumerate(combination)]
        binomes = itertools.combinations(combiWithView, 2)
        nbBinomes = int(math.factorial(nbViews) / 2 / math.factorial(nbViews-2))
        couple_diversities = np.zeros(nbBinomes)
        for binomeIndex, binome in enumerate(binomes):
            (viewIndex1, classifierIndex1), (viewIndex2, classifierIndex2) = binome
            folds_couple_diversity = np.mean(measurement(classifiersDecisions[viewIndex1, classifierIndex1],
                                               classifiersDecisions[viewIndex2, classifierIndex2], foldsGroudTruth)
                                , axis=1)
            couple_diversities[binomeIndex] = np.mean(folds_couple_diversity)
        div_measure[combinationsIndex] = np.mean(couple_diversities)
    bestCombiIndex = np.argmax(div_measure)
    bestCombination = combis[bestCombiIndex]

    return [classifiersNames[viewIndex][index] for viewIndex, index in enumerate(bestCombination)], div_measure[bestCombiIndex]


def global_div_measure(allClassifersNames, viewsIndices, resultsMonoview, measurement, foldsGroudTruth):
    """
    This function is used to get the max of a diversity measurement,passed as an argument
    It generates all possible combinations to estimate the diversity on a combination
    The best combination is the one that maximize the measurement.
    """
    classifiersDecisions, classifiersNames = getClassifiersDecisions(allClassifersNames,
                                                                     viewsIndices,
                                                                     resultsMonoview)

    foldsLen = len(resultsMonoview[0][1][6][0])
    nbViews = len(viewsIndices)
    nbClassifiers = len(allClassifersNames)
    combinations = itertools.combinations_with_replacement(range(nbClassifiers), nbViews)
    nbCombinations = int(math.factorial(nbClassifiers + nbViews - 1) / math.factorial(nbViews) / math.factorial(
        nbClassifiers - 1))
    div_measure = np.zeros(nbCombinations)
    combis = np.zeros((nbCombinations, nbViews), dtype=int)
    for combinationsIndex, combination in enumerate(combinations):
        combis[combinationsIndex] = combination
        div_measure[combinationsIndex] = measurement(classifiersDecisions, combination, foldsGroudTruth, foldsLen)
    bestCombiIndex = np.argmax(div_measure)
    bestCombination = combis[bestCombiIndex]

    return [classifiersNames[viewIndex][index] for viewIndex, index in enumerate(bestCombination)], div_measure[
        bestCombiIndex]


def CQ_div_measure(allClassifersNames, viewsIndices, resultsMonoview, measurement, foldsGroudTruth):
    """
    This function is used to measure a pseudo-CQ measurement based on the minCq algorithm.
    It's a mix between couple_div_measure and global_div_measure that uses multiple measurements.
    """
    classifiersDecisions, classifiersNames = getClassifiersDecisions(allClassifersNames,
                                                                     viewsIndices,
                                                                     resultsMonoview)
    foldsLen = len(resultsMonoview[0][1][6][0])
    nbViews = len(viewsIndices)
    nbClassifiers = len(allClassifersNames)
    combinations = itertools.combinations_with_replacement(range(nbClassifiers), nbViews)
    nbCombinations = int(
        math.factorial(nbClassifiers + nbViews - 1) / math.factorial(nbViews) / math.factorial(nbClassifiers - 1))
    div_measure = np.zeros(nbCombinations)
    combis = np.zeros((nbCombinations, nbViews), dtype=int)

    for combinationsIndex, combination in enumerate(combinations):
        combis[combinationsIndex] = combination
        combiWithView = [(viewIndex, combiIndex) for viewIndex, combiIndex in enumerate(combination)]
        binomes = itertools.combinations(combiWithView, 2)
        nbBinomes = int(math.factorial(nbViews) / 2 / math.factorial(nbViews - 2))
        disagreement = np.zeros(nbBinomes)
        div_measure[combinationsIndex] = measurement[1](classifiersDecisions, combination, foldsGroudTruth, foldsLen)
        for binomeIndex, binome in enumerate(binomes):
            (viewIndex1, classifierIndex1), (viewIndex2, classifierIndex2) = binome
            nbDisagree = np.sum(measurement[0](classifiersDecisions[viewIndex1, classifierIndex1],
                                            classifiersDecisions[viewIndex2, classifierIndex2], foldsGroudTruth)
                                , axis=1) / float(foldsLen)
            disagreement[binomeIndex] = np.mean(nbDisagree)
        div_measure[combinationsIndex] /= float(np.mean(disagreement))
    bestCombiIndex = np.argmin(div_measure)
    bestCombination = combis[bestCombiIndex]

    return [classifiersNames[viewIndex][index] for viewIndex, index in enumerate(bestCombination)], div_measure[
        bestCombiIndex]


def getFoldsGroundTruth(directory):
    """This function is used to get the labels of each fold example used in the measurements
    foldsGroundTruth is formatted as
    foldsGroundTruth[foldIndex, exampleIndex]"""
    foldsFilesNames = os.listdir(directory+"folds/")
    foldLen = len(np.genfromtxt(directory+"folds/"+foldsFilesNames[0], delimiter=','))
    foldsGroudTruth = np.zeros((len(foldsFilesNames), foldLen), dtype=int)
    for fileName in foldsFilesNames:
        foldIndex = int(fileName[-5])
        foldsGroudTruth[foldIndex] = np.genfromtxt(directory+"folds/"+fileName, delimiter=',')
    return foldsGroudTruth


def getArgs(args, benchmark, views, viewsIndices, randomState,
            directory, resultsMonoview, classificationIndices, measurement, name):
    """This function is a general function to get the args for all the measurements used"""
    foldsGroundTruth = getFoldsGroundTruth(directory)
    monoviewClassifierModulesNames = benchmark["Monoview"]
    if name in ['DisagreeFusion', 'DoubleFaultFusion']:
        classifiersNames, div_measure = couple_div_measure(monoviewClassifierModulesNames,
                                            viewsIndices, resultsMonoview, measurement, foldsGroundTruth)
    elif name == "PseudoCQFusion":
        classifiersNames, div_measure = CQ_div_measure(monoviewClassifierModulesNames,
                                            viewsIndices, resultsMonoview, measurement, foldsGroundTruth)
    else:
        classifiersNames, div_measure = global_div_measure(monoviewClassifierModulesNames,
                                            viewsIndices, resultsMonoview, measurement, foldsGroundTruth)
    multiclass_preds = [monoviewResult[1][5] for monoviewResult in resultsMonoview]
    if isBiclass(multiclass_preds):
        monoviewDecisions = np.array([monoviewResult[1][3] for monoviewResult in resultsMonoview
                                      if classifiersNames[viewsIndices.index(monoviewResult[0])] ==
                                                          monoviewResult[1][0]])
    else:
        monoviewDecisions = np.array(
            [genMulticlassMonoviewDecision(monoviewResult, classificationIndices) for monoviewResult in
             resultsMonoview if classifiersNames[viewsIndices.index(monoviewResult[0])] == monoviewResult[1][0]])
    argumentsList = []
    arguments = {"CL_type": name,
                 "views": views,
                 "NB_VIEW": len(views),
                 "viewsIndices": viewsIndices,
                 "NB_CLASS": len(args.CL_classes),
                 "LABELS_NAMES": args.CL_classes,
                 name+"KWARGS": {
                     "weights": args.DGF_weights,
                     "classifiersNames": classifiersNames,
                     "monoviewDecisions": monoviewDecisions,
                     "nbCLass": len(args.CL_classes),
                     "div_measure": div_measure
                 }
                 }
    argumentsList.append(arguments)
    return argumentsList

def genParamsSets(classificationKWARGS, randomState, nIter=1):
    """Used to generate parameters sets for the random hyper parameters optimization function"""
    weights = [randomState.random_sample(len(classificationKWARGS["classifiersNames"])) for _ in range(nIter)]
    nomralizedWeights = [[weightVector/np.sum(weightVector)] for weightVector in weights]
    return nomralizedWeights


class DiversityFusionClass:

    """This is a parent class for all the diversity fusion based classifiers."""

    def __init__(self, randomState, NB_CORES=1, **kwargs):
        """Used to init the instances"""
        if kwargs["weights"] == []:
            self.weights = [1.0/len(kwargs["classifiersNames"]) for _ in range(len(kwargs["classifiersNames"]))]
        else:
            self.weights = np.array(kwargs["weights"])/np.sum(np.array(kwargs["weights"]))
        self.monoviewDecisions = kwargs["monoviewDecisions"]
        self.classifiersNames = kwargs["classifiersNames"]
        self.nbClass = kwargs["nbCLass"]
        self.div_measure = kwargs["div_measure"]

    def setParams(self, paramsSet):
        """ Used to set the weights"""
        self.weights = paramsSet[0]

    def fit_hdf5(self, DATASET, labels, trainIndices=None, viewsIndices=None, metric=["f1_score", None]):
        """No need to fit as the monoview classifiers are already fitted"""
        pass

    def predict_hdf5(self, DATASET, usedIndices=None, viewsIndices=None):
        """Just a weighted majority vote"""
        if usedIndices is None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        votes = np.zeros((len(usedIndices), self.nbClass), dtype=float)
        for usedIndex, exampleIndex in enumerate(usedIndices):
            for monoviewDecisionIndex, monoviewDecision in enumerate(self.monoviewDecisions):
                votes[usedIndex, monoviewDecision[exampleIndex]] += 1#self.weights[monoviewDecisionIndex]
        predictedLabels = np.argmax(votes, axis=1)
        return predictedLabels

    def predict_probas_hdf5(self, DATASET, usedIndices=None):
        pass

    def getConfigString(self, classificationKWARGS):
        return "weights : "+", ".join(map(str, list(self.weights)))

    def getSpecificAnalysis(self, classificationKWARGS):
        stringAnalysis = "Classifiers used for each view : " + ', '.join(self.classifiersNames)
        return stringAnalysis

