from multiview_platform.MonoMultiViewClassifiers.Multiview.Additions import diversity_utils
from ..DifficultyFusion.DifficultyFusionModule import difficulty
from ..DoubleFaultFusion.DoubleFaultFusionModule import doubleFault


def genName(config):
    return "PseudoCQFusion"


def getBenchmark(benchmark, args=None):
    benchmark["Multiview"]["PseudoCQFusion"] = ["take_everything"]
    return benchmark


def pseudoCQ(difficulty, doubleFlaut):
    return difficulty/float(doubleFlaut)


def getArgs(args, benchmark, views, viewsIndices, randomState, directory, resultsMonoview, classificationIndices):
    return diversity_utils.getArgs(args, benchmark, views,
                                   viewsIndices, randomState, directory,
                                   resultsMonoview, classificationIndices,
                                   [doubleFault, difficulty], "PseudoCQFusion")


def genParamsSets(classificationKWARGS, randomState, nIter=1):
    return diversity_utils.genParamsSets(classificationKWARGS, randomState, nIter=nIter)



class PseudoCQFusionClass(diversity_utils.DiversityFusionClass):

    def __init__(self, randomState, NB_CORES=1, **kwargs):
        diversity_utils.DiversityFusionClass.__init__(self, randomState, NB_CORES=1, **kwargs)

    def getSpecificAnalysis(self, classificationKWARGS):
        stringAnalysis = "Classifiers used for each view : "+ ', '.join(self.classifiersNames)+\
                         ', with a pseudo CQ of '+str(self.div_measure)
        return stringAnalysis