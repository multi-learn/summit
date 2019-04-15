from sklearn.tree import DecisionTreeClassifier

from .AdaboostPregen import AdaboostPregen

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


class AdaboostPregen10(AdaboostPregen):

    def __init__(self, random_state=None, n_estimators=50,
                 base_estimator=None, n_stumps=1, self_complemeted=True,
                 **kwargs):
        super(AdaboostPregen10, self).__init__(
            random_state=random_state,
            n_estimators=100,
            base_estimator=base_estimator,
            n_stumps=10,
            self_complemeted=self_complemeted
        )


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {'n_estimators': args.AdP_n_est,
                  'base_estimator': DecisionTreeClassifier(max_depth=1),
                  }
    return kwargsDict


def paramsToSet(nIter, random_state):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"n_estimators": random_state.randint(1, 500),
                          "base_estimator": None})
    return paramsSet
