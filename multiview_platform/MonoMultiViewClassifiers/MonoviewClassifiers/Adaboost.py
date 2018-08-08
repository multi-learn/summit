from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from ..Monoview.MonoviewUtils import CustomRandint, BaseMonoviewClassifier

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


class Adaboost(AdaBoostClassifier, BaseMonoviewClassifier):

    def __init__(self, random_state=None, n_estimators=10,
                 base_estimator=None, **kwargs):
        super(Adaboost, self).__init__(
            random_state=random_state,
            n_estimators=n_estimators,
            base_estimator=base_estimator,
            )
        self.param_names = ["n_estimators", "base_estimator"]
        self.classed_params = ["base_estimator"]
        self.distribs = [CustomRandint(low=1, high=500), [None]]
        self.weird_strings = {"base_estimator":"class_name"}

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def paramsToSrt(self, nIter=1):
        """Used for weighted linear early fusion to generate random search sets"""
        paramsSet = []
        for _ in range(nIter):
            paramsSet.append({"n_estimators": self.random_state.randint(1, 150),
                              "base_estimator": None})
        return paramsSet

    def getInterpret(self, directory):
        interpretString = ""
        interpretString += self.getFeatureImportance(directory)
        return interpretString


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {'n_estimators': args.Ada_n_est,
                  'base_estimator': DecisionTreeClassifier()}
    return kwargsDict
# def canProbas():
#     return True
#
#
# def fit(DATASET, CLASS_LABELS, randomState, NB_CORES=1, **kwargs):
#     """Used to fit the monoview classifier with the args stored in kwargs"""
#     classifier = AdaBoostClassifier(n_estimators=kwargs['n_estimators'],
#                                     base_estimator=kwargs['base_estimator'],
#                                     random_state=randomState)
#     classifier.fit(DATASET, CLASS_LABELS)
#     return classifier
#
#
# def paramsToSet(nIter, randomState):
#     """Used for weighted linear early fusion to generate random search sets"""
#     paramsSet = []
#     for _ in range(nIter):
#         paramsSet.append({"n_estimators": randomState.randint(1, 15),
#                           "base_estimator": DecisionTreeClassifier()})
#     return paramsSet
#
#
# def getKWARGS(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {}
#     kwargsDict['n_estimators'] = args.Ada_n_est
#     kwargsDict['base_estimator'] = DecisionTreeClassifier() #args.Ada_b_est
#     return kwargsDict
#
#
# def genPipeline():
#     return Pipeline([('classifier', AdaBoostClassifier())])
#
#
# def genParamsDict(randomState):
#     return {"classifier__n_estimators": np.arange(150)+1,
#              "classifier__base_estimator": [DecisionTreeClassifier()]}
#
#
# def genBestParams(detector):
#     return {"n_estimators": detector.best_params_["classifier__n_estimators"],
#             "base_estimator": detector.best_params_["classifier__base_estimator"]}
#
#
# def genParamsFromDetector(detector):
#     nIter = len(detector.cv_results_['param_classifier__n_estimators'])
#     return [("baseEstimators", np.array(["DecisionTree" for _ in range(nIter)])),
#               ("nEstimators", np.array(detector.cv_results_['param_classifier__n_estimators']))]
#
#
# def getConfig(config):
#     if type(config) is not dict:  # Used in late fusion when config is a classifier
#         return "\n\t\t- Adaboost with num_esimators : " + str(config.n_estimators) + ", base_estimators : " + str(
#             config.base_estimator)
#     else:
#         return "\n\t\t- Adaboost with n_estimators : " + str(config["n_estimators"]) + ", base_estimator : " + str(
#                config["base_estimator"])
#
#
# def getInterpret(classifier, directory):
#     interpretString = getFeatureImportance(classifier, directory)
#     return interpretString