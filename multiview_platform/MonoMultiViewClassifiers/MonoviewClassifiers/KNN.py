from sklearn.neighbors import KNeighborsClassifier

from ..Monoview.MonoviewUtils import CustomRandint, BaseMonoviewClassifier

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


class KNN(KNeighborsClassifier, BaseMonoviewClassifier):

    def __init__(self, random_state=None, n_neighbors=5,
                 weights='uniform', algorithm='auto', p=2, **kwargs):
        super(KNN, self).__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            p=p
        )
        self.param_names = ["n_neighbors", "weights", "algorithm", "p",
                            "random_state", ]
        self.classed_params = []
        self.distribs = [CustomRandint(low=1, high=10), ["uniform", "distance"],
                         ["auto", "ball_tree", "kd_tree", "brute"], [1, 2],
                         [random_state]]
        self.weird_strings = {}
        self.random_state = random_state

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def getInterpret(self, directory, y_test):
        interpretString = ""
        return interpretString


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {"n_neighbors": args.KNN_neigh,
                  "weights": args.KNN_weights,
                  "algorithm": args.KNN_algo,
                  "p": args.KNN_p}
    return kwargsDict


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"n_neighbors": randomState.randint(1, 20),
                          "weights": randomState.choice(
                              ["uniform", "distance"]),
                          "algorithm": randomState.choice(
                              ["auto", "ball_tree", "kd_tree", "brute"]),
                          "p": randomState.choice([1, 2])})
    return paramsSet
