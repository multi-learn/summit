import numpy as np

from .. import diversity_utils


def genName(config):
    return "DoubleFaultFusion"


def getBenchmark(benchmark, args=None):
    benchmark["Multiview"]["DoubleFaultFusion"] = ["take_everything"]
    return benchmark


def doubleFault(classifierDecision1, classifierDecision2, ground_truth):
    return np.logical_and(np.logical_not(np.logical_and(classifierDecision1, ground_truth)),
                          np.logical_not(np.logical_and(classifierDecision2, ground_truth)))


def getArgs(args, benchmark, views, viewsIndices, randomState, directory, resultsMonoview, classificationIndices):
    return diversity_utils.getArgs(args, benchmark, views,
                                   viewsIndices, randomState, directory,
                                   resultsMonoview, classificationIndices,
                                   doubleFault, "DoubleFaultFusion")


def genParamsSets(classificationKWARGS, randomState, nIter=1):
    return diversity_utils.genParamsSets(classificationKWARGS, randomState, nIter=nIter)



class DoubleFaultFusionClass(diversity_utils.DiversityFusionClass):

    def __init__(self, randomState, NB_CORES=1, **kwargs):
        diversity_utils.DiversityFusionClass.__init__(self, randomState, NB_CORES=1, **kwargs)

    def getSpecificAnalysis(self, classificationKWARGS):
        stringAnalysis = "Classifiers used for each view : "+ ', '.join(self.classifiersNames)+\
                         ', with a double fault ratio of '+str(self.div_measure)
        return stringAnalysis