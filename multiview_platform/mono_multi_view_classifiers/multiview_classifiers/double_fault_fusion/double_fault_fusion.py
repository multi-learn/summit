import numpy as np

from ...multiview.additions import diversity_utils


def genName(config):
    return "double_fault_fusion"


def getBenchmark(benchmark, args=None):
    benchmark["multiview"]["double_fault_fusion"] = ["take_everything"]
    return benchmark


def doubleFault(classifierDecision1, classifierDecision2, ground_truth):
    return np.logical_and(np.logical_xor(classifierDecision1, ground_truth),
                          np.logical_xor(classifierDecision2, ground_truth))


def getArgs(args, benchmark, views, views_indices, random_state, directory, resultsMonoview, classificationIndices):
    return diversity_utils.getArgs(args, benchmark, views,
                                   views_indices, random_state, directory,
                                   resultsMonoview, classificationIndices,
                                   doubleFault, "double_fault_fusion")


def genParamsSets(classificationKWARGS, random_state, nIter=1):
    return diversity_utils.genParamsSets(classificationKWARGS, random_state, nIter=nIter)



class DoubleFaultFusionClass(diversity_utils.DiversityFusionClass):

    def __init__(self, random_state, NB_CORES=1, **kwargs):
        diversity_utils.DiversityFusionClass.__init__(self, random_state, NB_CORES=1, **kwargs)

    def getSpecificAnalysis(self, classificationKWARGS):
        stringAnalysis = "Classifiers used for each view : "+ ', '.join(self.classifiers_names)+\
                         ', with a double fault ratio of '+str(self.div_measure)
        return stringAnalysis