from sklearn.externals.six import iteritems
from pyscm.scm import SetCoveringMachineClassifier as scm
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

from ..Monoview.MonoviewUtils import CustomRandint, CustomUniform, BaseMonoviewClassifier, change_label_to_minus, change_label_to_zero
from ..Monoview.Additions.BoostUtils import StumpsClassifiersGenerator, BaseBoost
from ..Monoview.Additions.PregenUtils import PregenClassifier
# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

class SCMPregen(scm, BaseMonoviewClassifier, PregenClassifier):

    def __init__(self, random_state=None, model_type="conjunction",
                 max_rules=10, p=0.1, n_stumps=10,self_complemented=True, **kwargs):
        super(SCMPregen, self).__init__(
            random_state=random_state,
            model_type=model_type,
            max_rules=max_rules,
            p=p
            )
        self.param_names = ["model_type", "max_rules", "p", "n_stumps", "random_state"]
        self.distribs = [["conjunction", "disjunction"],
                         CustomRandint(low=1, high=15),
                         CustomUniform(loc=0, state=1), [n_stumps], [random_state]]
        self.classed_params = []
        self.weird_strings = {}
        self.self_complemented = self_complemented
        self.n_stumps = n_stumps
        self.estimators_generator = None

    def fit(self, X, y, tiebreaker=None, iteration_callback=None, **fit_params):
        pregen_X, pregen_y = self.pregen_voters(X, y)
        super(SCMPregen, self).fit(pregen_X, pregen_y)
        return self

    def predict(self, X):
        pregen_h, _ = self.pregen_voters(X)
        from time import sleep;sleep(1)
        return self.classes_[self.model_.predict(X)]

    def get_params(self, deep=True):
        return {"p": self.p, "model_type": self.model_type,
         "max_rules": self.max_rules,
         "random_state": self.random_state, "n_stumps":self.n_stumps}

    # def pregen_voters(self, X, y=None):
    #     if y is not None:
    #         if self.estimators_generator is None:
    #             self.estimators_generator = StumpsClassifiersGenerator(
    #                 n_stumps_per_attribute=self.n_stumps,
    #                 self_complemented=self.self_complemented)
    #         self.estimators_generator.fit(X, y)
    #     else:
    #         neg_y=None
    #     classification_matrix = self._binary_classification_matrix_t(X)
    #     return classification_matrix, y
    #
    # def _collect_probas_t(self, X):
    #     print('jb')
    #     for est in self.estimators_generator.estimators_:
    #         print(type(est))
    #         print(est.predict_proba_t(X))
    #     print('ha')
    #     return np.asarray([clf.predict_proba(X) for clf in self.estimators_generator.estimators_])
    #
    # def _binary_classification_matrix_t(self, X):
    #     probas = self._collect_probas_t(X)
    #     predicted_labels = np.argmax(probas, axis=2)
    #     predicted_labels[predicted_labels == 0] = -1
    #     values = np.max(probas, axis=2)
    #     return (predicted_labels * values).T


    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return False

    def getInterpret(self, directory, y_test):
        interpretString = "Model used : " + str(self.model_)
        return interpretString


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {"model_type": args.SCP_model_type,
                  "p": args.SCP_p,
                  "max_rules": args.SCP_max_rules,
                  "n_stumps": args.SCP_stumps}
    return kwargsDict


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"model_type": randomState.choice(["conjunction", "disjunction"]),
                          "max_rules": randomState.randint(1, 15),
                          "p": randomState.random_sample()})
    return paramsSet


