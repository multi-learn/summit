from MKLpy.algorithms import EasyMKL
from MKLpy.metrics import pairwise
from MKLpy.lists import HPK_generator
from MKLpy.algorithms.komd import KOMD
import numpy as np

from ..multiview.multiview_utils import BaseMultiviewClassifier, get_examples_views_indices
from ..utils.hyper_parameter_search import CustomUniform


classifier_class_name = "EasyMKLClassifier"

class EasyMKLClassifier(BaseMultiviewClassifier, EasyMKL):

    def __init__(self, random_state=None, degrees=1, lam=0.1,
                 learner=KOMD(lam=0.1), generator=HPK_generator(n=10),
                 multiclass_strategy='ova', verbose=False):
        super().__init__(random_state)
        super(BaseMultiviewClassifier, self).__init__(lam=lam,
                                                      learner=learner,
                                                      generator=generator,
                                                      multiclass_strategy=multiclass_strategy,
                                                      verbose=verbose)
        self.degrees = degrees
        self.param_names = ["lam", "degrees"]
        self.distribs = [CustomUniform(), DegreesGenerator()]

    def fit(self, X, y, train_indices=None, view_indices=None ):
        train_indices, view_indices = get_examples_views_indices(X,
                                                                  train_indices,
                                                                  view_indices)
        if isinstance(self.degrees, DegreesDistribution):
            self.degrees = self.degrees.draw(len(view_indices))
        elif isinstance(int, self.degrees):
            self.degrees = [self.degrees for _ in range(len(view_indices))]

        kernels = [pairwise.homogeneous_polynomial_kernel(X.get_v(view_indices[index],
                                                                  train_indices),
                                                          degree=degree)
                   for index, degree in enumerate(self.degrees)]
        return super(EasyMKLClassifier, self).fit(kernels, y[train_indices])

    def predict(self, X, example_indices=None, view_indices=None):
        example_indices, view_indices = get_examples_views_indices(X,
                                                                  example_indices,
                                                                  view_indices)
        kernels = [
            pairwise.homogeneous_polynomial_kernel(X.get_v(view_indices[index],
                                                           example_indices),
                                                   degree=degree)
            for index, degree in enumerate(self.degrees)]
        return super(EasyMKLClassifier, self).predict(kernels,)


class DegreesGenerator:

    def __init__(self):
        pass

    def rvs(self, random_state=None):
        return DegreesDistribution(seed=random_state.randint(1))


class DegreesDistribution:

    def __init__(self, seed=42):
        self.random_state=np.random.RandomState(seed)

    def draw(self, nb_view):
        return self.random_state.randint(low=5,high=10,size=nb_view)
