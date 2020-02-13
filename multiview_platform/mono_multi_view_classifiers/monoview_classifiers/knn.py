from sklearn.neighbors import KNeighborsClassifier

from ..monoview.monoview_utils import CustomRandint, BaseMonoviewClassifier

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


classifier_class_name = "KNN"


class KNN(KNeighborsClassifier, BaseMonoviewClassifier):
    """
    Implement extention of KNeighborsClassifier of sklearn
    for the usage of the multiview_platform.

    Parameters
    ----------
    random_state
    n_neighbors
    weights
    algorithm
    p
    kwargs
    """
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

    # def canProbas(self):
    #     """Used to know if the classifier can return label probabilities"""
    #     return True

    def get_interpretation(self, directory, y_test):
        interpretString = ""
        return interpretString


# def formatCmdArgs(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {"n_neighbors": args.KNN_neigh,
#                   "weights": args.KNN_weights,
#                   "algorithm": args.KNN_algo,
#                   "p": args.KNN_p}
#     return kwargsDict


def paramsToSet(nIter, random_state):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"n_neighbors": random_state.randint(1, 20),
                          "weights": random_state.choice(
                              ["uniform", "distance"]),
                          "algorithm": random_state.choice(
                              ["auto", "ball_tree", "kd_tree", "brute"]),
                          "p": random_state.choice([1, 2])})
    return paramsSet
