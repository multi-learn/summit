from sklearn.linear_model import SGDClassifier

from ..monoview.monoview_utils import BaseMonoviewClassifier
from summit.multiview_platform.utils.hyper_parameter_search import CustomUniform

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "SGD"


class SGD(SGDClassifier, BaseMonoviewClassifier):
    """
    This class is an adaptation of scikit-learn's `SGDClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html>`_


    """

    def __init__(self, random_state=None, loss='hinge',
                 penalty='l2', alpha=0.0001, max_iter=5, tol=None, **kwargs):
        SGDClassifier.__init__(self,
                               loss=loss,
                               penalty=penalty,
                               alpha=alpha,
                               max_iter=5,
                               tol=None,
                               random_state=random_state
                               )
        self.param_names = ["loss", "penalty", "alpha", "random_state"]
        self.classed_params = []
        self.distribs = [['log', 'modified_huber'],
                         ["l1", "l2", "elasticnet"],
                         CustomUniform(loc=0, state=1), [random_state]]
        self.weird_strings = {}
