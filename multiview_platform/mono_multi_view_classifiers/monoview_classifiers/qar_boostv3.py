from ..monoview.additions.CGDescUtils import ColumnGenerationClassifierQar
from ..monoview.monoview_utils import BaseMonoviewClassifier


class QarBoostv3(ColumnGenerationClassifierQar, BaseMonoviewClassifier):

    def __init__(self, random_state=None, **kwargs):
        super(QarBoostv3, self).__init__(
            random_state=random_state,
            self_complemented=False,
            twice_the_same=False,
            c_bound_choice=True,
            random_start=True,
            n_stumps=1,
            use_r=False
        )
        # old_fashioned=False
        # previous_vote_weighted=False,
        # two_wieghts_problem=False,
        # divided_ponderation=True,
        # n_stumps_per_attribute=1,
        self.param_names = []
        self.classed_params = []
        self.distribs = []
        self.weird_strings = {}

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def getInterpret(self, directory, y_test):
        return self.getInterpretQar(directory, y_test)

    def get_name_for_fusion(self):
        return "QBv3"


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {}
    return kwargsDict


def paramsToSet(nIter, randomState):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({})
    return paramsSet
