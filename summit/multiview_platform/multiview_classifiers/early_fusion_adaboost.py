from .additions.early_fusion_from_monoview import BaseEarlyFusion
from ..utils.hyper_parameter_search import CustomRandint
from ..utils.base import base_boosting_estimators

# from ..utils.dataset import get_v

classifier_class_name = "EarlyFusionAdaboost"


class EarlyFusionAdaboost(BaseEarlyFusion):

    def __init__(self, random_state=None, n_estimators=50,
                 estimator=None, base_estimator_config=None, **kwargs):
        BaseEarlyFusion.__init__(self, random_state=random_state,
                                 monoview_classifier="adaboost",
                                 n_estimators= n_estimators,
                                 estimator=estimator,
                                 base_estimator_config=base_estimator_config, **kwargs)
        self.param_names = ["n_estimators", "estimator"]
        self.classed_params = ["estimator"]
        self.distribs = [CustomRandint(low=1, high=500),
                         base_boosting_estimators]
        self.weird_strings = {"estimator": "class_name"}
