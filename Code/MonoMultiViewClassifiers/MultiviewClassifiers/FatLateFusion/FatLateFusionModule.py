import numpy as np




def genName(config):
    return "FatLateFusion"

def getBenchmark(benchmark, args=None):
    benchmark["Multiview"]["FatLateFusion"] = ["take_everything"]
    return benchmark


def getArgs(args, benchmark, views, viewsIndices, randomState, directory, resultsMonoview, classificationIndices):
    argumentsList = []
    monoviewDecisions = np.array([monoviewResult[1][3] for monoviewResult in resultsMonoview])
    arguments = {"CL_type": "FatLateFusion",
                 "views": ["all"],
                 "NB_VIEW": len(resultsMonoview),
                 "viewsIndices": range(len(resultsMonoview)),
                 "NB_CLASS": len(args.CL_classes),
                 "LABELS_NAMES": args.CL_classes,
                 "FatLateFusionKWARGS": {
                     "monoviewDecisions": monoviewDecisions
                 }
                 }
    argumentsList.append(arguments)
    return argumentsList


def genParamsSets(classificationKWARGS, randomState, nIter=1):
    """Used to generate parameters sets for the random hyper parameters optimization function"""
    nbMonoviewClassifiers = len(classificationKWARGS["monoviewDecisions"])
    weights = [randomState.random_sample(nbMonoviewClassifiers) for _ in range(len(classificationKWARGS["monoviewDecisions"]))]
    nomralizedWeights = [weights/np.sum(weights)]
    return nomralizedWeights

class FatLateFusionClass:

    def __init__(self, randomState, NB_CORES=1, **kwargs):
        self.weights = [1.0/len(["monoviewDecisions"]) for _ in range(len(["monoviewDecisions"]))]
        self.monoviewDecisions = kwargs["monoviewDecisions"]

    def setParams(self, paramsSet):
        self.weights = paramsSet[0]

    def fit_hdf5(self, DATASET, labels, trainIndices=None, viewsIndices=None, metric=["f1_score", None]):
        pass

    def predict_hdf5(self, DATASET, usedIndices=None, viewsIndices=None):
        if usedIndices is None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        votes = np.zeros((len(usedIndices), DATASET.get("Metadata").attrs["nbClass"]), dtype=float)
        for usedIndex, exampleIndex in enumerate(usedIndices):
            for monoviewDecisionIndex, monoviewDecision in enumerate(self.monoviewDecisions):
                votes[usedIndex, monoviewDecision[exampleIndex]] += self.weights[monoviewDecisionIndex]
        predictedLabels = np.argmax(votes, axis=1)
        return predictedLabels

    def predict_probas_hdf5(self, DATASET, usedIndices=None):
        pass
        # if usedIndices is None:
        #     usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        # votes = np.zeros((DATASET.get("Metadata").attrs["datasetLength"], DATASET.get("Metadata").attrs["nbClass"]), dtype=float)
        # for exampleIndex in usedIndices:
        #     for monoviewDecisionIndex, monoviewDecision in enumerate(self.monoviewDecisions):
        #         votes[exampleIndex, monoviewDecision[exampleIndex]] += self.weights[monoviewDecisionIndex]
        # predictedProbas =
        # return predictedLabels
