from .additions.early_fusion_from_monoview import BaseEarlyFusion
from ..utils.hyper_parameter_search import CustomRandint
from ..monoview_classifiers.gradient_boosting import CustomDecisionTreeGB

classifier_class_name = "EarlyFusionGB"


class EarlyFusionGB(BaseEarlyFusion):

    def __init__(self, random_state=None, loss="exponential", max_depth=1.0,
                 n_estimators=100,
                 init=CustomDecisionTreeGB(max_depth=1),
                 **kwargs):
        BaseEarlyFusion.__init__(self, random_state=random_state,
                                 monoview_classifier="gradient_boosting",
                                 loss=loss, max_depth=max_depth,
                                 n_estimators=n_estimators, init=init, **kwargs)
        self.param_names = ["n_estimators", "max_depth"]
        self.classed_params = []
        self.distribs = [CustomRandint(low=50, high=500),
                         CustomRandint(low=1, high=10), ]