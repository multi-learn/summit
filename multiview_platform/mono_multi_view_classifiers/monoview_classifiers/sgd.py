from sklearn.linear_model import SGDClassifier

from ..monoview.monoview_utils import CustomUniform, BaseMonoviewClassifier

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "SGD"


class SGD(SGDClassifier, BaseMonoviewClassifier):
    """

    Parameters
    ----------
    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number multiview_generator to use when
        shuffling the data.

    loss : str , (default = "hinge")
    penalty : str, (default = "l2")

    alpha : float, (default = 0.0001)

    kwargs : other arguments


    Attributes
    ----------
    param_names :

    distribs :

    classed_params :

    weird_strings :

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
