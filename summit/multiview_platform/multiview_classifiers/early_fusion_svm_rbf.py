from .additions.early_fusion_from_monoview import BaseEarlyFusion
from ..utils.hyper_parameter_search import CustomUniform

classifier_class_name = "EarlyFusionSVMRBF"


class EarlyFusionSVMRBF(BaseEarlyFusion):

    def __init__(self, random_state=None, C=1.0, **kwargs):
        BaseEarlyFusion.__init__(self, random_state=random_state,
                                 monoview_classifier="svm_rbf", C=C, **kwargs)
        self.param_names = ["C", "random_state"]
        self.distribs = [CustomUniform(loc=0, state=1), [random_state]]