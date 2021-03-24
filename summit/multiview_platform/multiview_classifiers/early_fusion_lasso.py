from .additions.early_fusion_from_monoview import BaseEarlyFusion
from ..utils.hyper_parameter_search import CustomUniform, CustomRandint

classifier_class_name = "EarlyFusionLasso"


class EarlyFusionLasso(BaseEarlyFusion):

    def __init__(self, random_state=None, alpha=1.0,
                 max_iter=10, warm_start=False, **kwargs):
        BaseEarlyFusion.__init__(self, random_state=None, alpha=alpha,
                                 max_iter=max_iter,
                                 warm_start=warm_start, **kwargs)
        self.param_names = ["max_iter", "alpha", "random_state"]
        self.classed_params = []
        self.distribs = [CustomRandint(low=1, high=30--0),
                         CustomUniform(), [random_state]]