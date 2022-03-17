from .additions.early_fusion_from_monoview import BaseEarlyFusion
from ..utils.hyper_parameter_search import CustomRandint

# from ..utils.dataset import get_v

classifier_class_name = "EarlyFusionDT"


class EarlyFusionDT(BaseEarlyFusion):

    def __init__(self, random_state=None, max_depth=None,
                 criterion='gini', splitter='best', **kwargs):
        BaseEarlyFusion.__init__(self, random_state=random_state,
                                 monoview_classifier="decision_tree", max_depth=max_depth,
                                 criterion=criterion, splitter=splitter, **kwargs)
        self.param_names = ["max_depth", "criterion", "splitter",
                            'random_state']
        self.classed_params = []
        self.distribs = [CustomRandint(low=1, high=300),
                         ["gini", "entropy"],
                         ["best", "random"], [random_state]]
        self.weird_strings = {}