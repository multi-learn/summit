import numpy as np

from .. import diversity_utils


def genName(config):
    return "DifficultyFusion"


def getBenchmark(benchmark, args=None):
    benchmark["Multiview"]["DifficultyFusion"] = ["take_everything"]
    return benchmark


def difficulty(classifiersDecisions, combination, foldsGroudTruth, foldsLen):
    nbView, _, nbFolds, nbExamples = classifiersDecisions.shape
    scores = np.zeros((nbView, nbFolds, nbExamples), dtype=int)
    for viewIndex, classifierIndex in enumerate(combination):
        scores[viewIndex] = np.logical_not(
            np.logical_xor(classifiersDecisions[viewIndex, classifierIndex],
                           foldsGroudTruth)
        )
    difficulty_scores = np.sum(scores, axis=0)
    difficulty_score = np.mean(
        np.var(
            np.array([
                         np.sum((difficulty_scores==viewIndex), axis=1)
                         for viewIndex in range(len(combination)+1)])
            , axis=0)
    )
    return difficulty_score


def getArgs(args, benchmark, views, viewsIndices, randomState, directory, resultsMonoview, classificationIndices):
    return diversity_utils.getArgs(args, benchmark, views,
                                   viewsIndices, randomState, directory,
                                   resultsMonoview, classificationIndices,
                                   difficulty, "DifficultyFusion")


def genParamsSets(classificationKWARGS, randomState, nIter=1):
    return diversity_utils.genParamsSets(classificationKWARGS, randomState, nIter=nIter)



class DifficultyFusionClass(diversity_utils.DiversityFusionClass):

    def __init__(self, randomState, NB_CORES=1, **kwargs):
        diversity_utils.DiversityFusionClass.__init__(self, randomState, NB_CORES=1, **kwargs)

    def getSpecificAnalysis(self, classificationKWARGS):
        stringAnalysis = "Classifiers used for each view : "+ ', '.join(self.classifiersNames)+\
                         ', with a difficulty of '+str(self.div_measure)
        return stringAnalysis