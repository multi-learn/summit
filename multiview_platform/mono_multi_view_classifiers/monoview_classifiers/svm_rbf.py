from multiview_platform.mono_multi_view_classifiers.monoview_classifiers.additions.SVCClassifier import \
    SVCClassifier
from ..monoview.monoview_utils import CustomUniform, BaseMonoviewClassifier

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "SVMRBF"


class SVMRBF(SVCClassifier, BaseMonoviewClassifier):
    """
    class SVMRBF for classifier SVCC

    Parameters
    ----------
    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number multiview_generator to use when
        shuffling the data.

    C :

    kwargs : others arguments

    Attributes
    ----------

    param_names : list of parameters names

    distribs :  list of random_state distribution
    """

    def __init__(self, random_state=None, C=1.0, **kwargs):
        SVCClassifier.__init__(self,
                               C=C,
                               kernel='rbf',
                               random_state=random_state
                               )
        self.param_names = ["C", "random_state"]
        self.distribs = [CustomUniform(loc=0, state=1), [random_state]]
