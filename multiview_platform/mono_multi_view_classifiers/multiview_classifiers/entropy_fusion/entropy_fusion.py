import numpy as np

from ...multiview.additions import diversity_utils


def genName(config):
    return "entropy_fusion"


def getBenchmark(benchmark, args=None):
    benchmark["multiview"]["entropy_fusion"] = ["take_everything"]
    return benchmark


def entropy(classifiersDecisions, combination, foldsGroudTruth, foldsLen):
    nbView, _, nbFolds, nbExamples = classifiersDecisions.shape
    scores = np.zeros((nbView, nbFolds, nbExamples), dtype=int)
    for viewIndex, classifierIndex in enumerate(combination):
        scores[viewIndex] = np.logical_not(
            np.logical_xor(classifiersDecisions[viewIndex, classifierIndex],
                           foldsGroudTruth)
        )
    entropy_scores = np.sum(scores, axis=0)
    nbViewMatrix = np.zeros((nbFolds, nbExamples), dtype=int)+nbView-entropy_scores
    entropy_score = np.mean(np.mean(np.minimum(entropy_scores, nbViewMatrix).astype(float)/(nbView - int(nbView/2)), axis=1))
    return entropy_score


def getArgs(args, benchmark, views, views_indices, randomState, directory, resultsMonoview, classificationIndices):
    return diversity_utils.getArgs(args, benchmark, views,
                                   views_indices, randomState, directory,
                                   resultsMonoview, classification_indices,
                                   entropy, "entropy_fusion")


def genParamsSets(classificationKWARGS, randomState, nIter=1):
    return diversity_utils.genParamsSets(classificationKWARGS, randomState, nIter=nIter)



class EntropyFusionClass(diversity_utils.DiversityFusionClass):

    def __init__(self, randomState, NB_CORES=1, **kwargs):
        diversity_utils.DiversityFusionClass.__init__(self, randomState, NB_CORES=1, **kwargs)

    def getSpecificAnalysis(self, classificationKWARGS):
        stringAnalysis = "Classifiers used for each view : "+ ', '.join(self.classifiers_names)+\
                         ', with an entropy of '+str(self.div_measure)
        return stringAnalysis