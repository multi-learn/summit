from sklearn.svm import SVC

from .additions.jumbo_fusion_utils import BaseJumboFusion
from ..monoview.monoview_utils import CustomUniform, CustomRandint

classifier_class_name = "SVMJumboFusion"

class SVMJumboFusion(BaseJumboFusion):

    def __init__(self, random_state=None, classifiers_names=None,
                 classifier_configs=None, nb_cores=1, weights=None, nb_monoview_per_view=1,  C=1.0, kernel="rbf", degree=2):
        self.need_probas=False
        super(SVMJumboFusion, self).__init__(random_state, classifiers_names=classifiers_names,
                                             classifier_configs=classifier_configs,
                                             nb_cores=nb_cores, weights=weights, nb_monoview_per_view=nb_monoview_per_view)
        self.param_names += ["C", "kernel", "degree"]
        self.distribs += [CustomUniform(), ["rbf", "poly", "linear"], CustomRandint(2, 5)]
        self.aggregation_estimator = SVC(C=C, kernel=kernel, degree=degree)

    def set_params(self, C=1.0, kernel="rbf", degree=1, **params):
        super(SVMJumboFusion, self).set_params(**params)
        self.aggregation_estimator.set_params(C=C, kernel=kernel, degree=degree)
        return self




