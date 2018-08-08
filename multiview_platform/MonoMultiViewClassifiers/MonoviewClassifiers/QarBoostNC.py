import numpy as np
from sklearn.pipeline import Pipeline
import time

from ..Monoview.Additions.BoostUtils import getInterpretBase
from ..Monoview.Additions.QarBoostUtils import QarBoostClassifier


class QarBoostNC(QarBoostClassifier):

    def __init__(self, random_state, **kwargs):
        super(QarBoostNC, self).__init__(
            mu=kwargs['mu'],
            epsilon=kwargs['epsilon'],
            n_max_iterations= kwargs['n_max_iterations'],
            random_state = random_state)

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return False

    def paramsToSrt(self, nIter=1):
        """Used for weighted linear early fusion to generate random search sets"""
        paramsSet = []
        for _ in range(nIter):
            paramsSet.append({"mu": 0.001,
                              "epsilon": 1e-08,
                              "n_max_iterations": None})
        return paramsSet

    def getKWARGS(self, args):
        """Used to format kwargs for the parsed args"""
        kwargsDict = {}
        kwargsDict['mu'] = 0.001
        kwargsDict['epsilon'] = 1e-08
        kwargsDict['n_max_iterations'] = None
        return kwargsDict

    def genPipeline(self):
        return Pipeline([('classifier', QarBoostClassifier())])

    def genParamsDict(self, randomState):
        return {"classifier__n_max_iterations": [None],
                "classifier__self_complemented": [False]}

    def genBestParams(self, detector):
        return {"n_max_iterations": detector.best_params_["classifier__n_max_iterations"]}

    def genParamsFromDetector(self, detector):
        nIter = len(detector.cv_results_['param_classifier__mu'])
        return [("mu", np.array([0.001 for _ in range(nIter)])),
                ("epsilon", np.array(detector.cv_results_['param_classifier__epsilon'])),
                ("n_max_iterations", np.array(detector.cv_results_['param_classifier__n_max_iterations']))]

    def getConfig(self, config):
        if type(config) is not dict:  # Used in late fusion when config is a classifier
            return "\n\t\t- QarBoost with mu : " + str(config.mu) + ", epsilon : " + str(
                config.epsilon + ", n_max_iterations : " + str(config.n_max_iterations))
        else:
            return "\n\t\t- QarBoost with mu : " + str(config["mu"]) + ", epsilon : " + str(
                   config["epsilon"] + ", n_max_iterations : " + str(config["n_max_iterations"]))

    def getInterpret(self, classifier, directory):
        interpretString = ""
        return interpretString




def canProbas():
    return False


def fit(DATASET, CLASS_LABELS, randomState, NB_CORES=1, **kwargs):
    start =time.time()
    """Used to fit the monoview classifier with the args stored in kwargs"""
    classifier = QarBoostClassifier(n_max_iterations=kwargs["n_max_iterations"],
                                      random_state=randomState,
                                      self_complemented=False)
    classifier.fit(DATASET, CLASS_LABELS)
    end = time.time()
    classifier.train_time = end-start
    return classifier


def paramsToSet(nIter, randomState):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"mu": randomState.uniform(1e-02, 10**(-0.5)),
                          "epsilon": 10**-randomState.randint(1, 15),
                          "n_max_iterations": None,
                          "self_complemented":False})
    return paramsSet


def getKWARGS(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {}
    kwargsDict['n_max_iterations'] = None
    return kwargsDict


def genPipeline():
    return Pipeline([('classifier', QarBoostClassifier())])


def genParamsDict(randomState):
    return {"classifier__n_max_iterations": [None],
            "classifier__random_state":[randomState],
            "classifier__self_complemented": [False]}


def genBestParams(detector):
    return {"n_max_iterations": detector.best_params_["classifier__n_max_iterations"]}


def genParamsFromDetector(detector):
    nIter = len(detector.cv_results_['param_classifier__mu'])
    return [("n_max_iterations", np.array(detector.cv_results_['param_classifier__n_max_iterations']))]


def getConfig(config):
    if type(config) is not dict:  # Used in late fusion when config is a classifier
        return "\n\t\t- QarBoostNC with n_max_iterations : " + str(config.n_max_iterations)
    else:
        return "\n\t\t- QarBoostNC with  n_max_iterations : " + str(config["n_max_iterations"])


def getInterpret(classifier, directory):
    return getInterpretBase(classifier, directory, "QarBoostNC", classifier.weights_, classifier.break_cause)

