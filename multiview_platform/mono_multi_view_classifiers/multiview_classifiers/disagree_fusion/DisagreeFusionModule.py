import numpy as np

from ...multiview.Additions import diversity_utils


def genName(config):
    return "disagree_fusion"


def getBenchmark(benchmark, args=None):
    benchmark["multiview"]["disagree_fusion"] = ["take_everything"]
    return benchmark


def disagree(classifierDecision1, classifierDecision2, ground_truth):
    return np.logical_xor(classifierDecision1, classifierDecision2)


def getArgs(args, benchmark, views, viewsIndices, randomState, directory, resultsMonoview, classificationIndices):
    return diversity_utils.getArgs(args, benchmark, views, viewsIndices,
                                   randomState, directory, resultsMonoview,
                                   classificationIndices, disagree, "disagree_fusion")


def genParamsSets(classificationKWARGS, randomState, nIter=1):
    return diversity_utils.genParamsSets(classificationKWARGS, randomState, nIter=nIter)
    # """Used to generate parameters sets for the random hyper parameters optimization function"""
    # weights = [randomState.random_sample(len(classificationKWARGS["classifiersNames"])) for _ in range(nIter)]
    # nomralizedWeights = [[weightVector/np.sum(weightVector)] for weightVector in weights]
    # return nomralizedWeights


class DisagreeFusionClass(diversity_utils.DiversityFusionClass):

    def __init__(self, randomState, NB_CORES=1, **kwargs):
        diversity_utils.DiversityFusionClass.__init__(self, randomState, NB_CORES=1, **kwargs)

    def getSpecificAnalysis(self, classificationKWARGS):
        stringAnalysis = "Classifiers used for each view : "+ ', '.join(self.classifiersNames)+\
                         ', with a disagreement of '+str(self.div_measure)
        return stringAnalysis
