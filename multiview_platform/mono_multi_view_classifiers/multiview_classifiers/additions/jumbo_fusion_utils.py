import numpy as np

from .late_fusion_utils import LateFusionClassifier
from ...monoview.monoview_utils import CustomRandint
from ...utils.dataset import get_examples_views_indices

class BaseJumboFusion(LateFusionClassifier):

    def __init__(self, random_state, classifiers_names=None,
                 classifier_configs=None,
                 nb_cores=1, weights=None, nb_monoview_per_view=1, rs=None):
        super(BaseJumboFusion, self).__init__(random_state, classifiers_names=classifiers_names,
                                             classifier_configs=classifier_configs,
                                             nb_cores=nb_cores, weights=weights,
                                              rs=rs)
        self.param_names += ["nb_monoview_per_view", ]
        self.distribs += [CustomRandint(1,10)]
        self.nb_monoview_per_view = nb_monoview_per_view

    def set_params(self, nb_monoview_per_view=1, **params):
        self.nb_monoview_per_view = nb_monoview_per_view
        super(BaseJumboFusion, self).set_params(**params)

    def predict(self, X, example_indices=None, view_indices=None):
        example_indices, view_indices = get_examples_views_indices(X, example_indices, view_indices)
        monoview_decisions = self.predict_monoview(X, example_indices=example_indices, view_indices=view_indices)
        return self.aggregation_estimator.predict(monoview_decisions)

    def fit(self, X, y, train_indices=None, view_indices=None):
        train_indices, view_indices = get_examples_views_indices(X, train_indices, view_indices)
        self.init_classifiers(len(view_indices), nb_monoview_per_view=self.nb_monoview_per_view)
        self.fit_monoview_estimators(X, y, train_indices=train_indices, view_indices=view_indices)
        monoview_decisions = self.predict_monoview(X, example_indices=train_indices, view_indices=view_indices)
        self.aggregation_estimator.fit(monoview_decisions, y[train_indices])
        return self

    def fit_monoview_estimators(self, X, y, train_indices=None, view_indices=None):
        self.monoview_estimators = [[self.init_monoview_estimator(classifier_name,
                                                                  self.classifier_configs[classifier_index])
                                     for classifier_index, classifier_name
                                     in enumerate(self.classifiers_names)]
                                    for _ in view_indices]

        self.monoview_estimators = [[estimator.fit(X.get_v(view_indices[idx], train_indices), y[train_indices])
                                     for estimator in view_estimators]
                                    for idx, view_estimators in enumerate(self.monoview_estimators)]
        return self

    def predict_monoview(self, X, example_indices=None, view_indices=None):
        monoview_decisions = np.zeros((len(example_indices), len(view_indices)*len(self.classifiers_names)))
        for idx, view_estimators in enumerate(self.monoview_estimators):
            for estimator_index, estimator in enumerate(view_estimators):
                monoview_decisions[:, len(self.classifiers_names)*idx+estimator_index] = estimator.predict(X.get_v(view_indices[idx],
                                                                                                                   example_indices))
        return monoview_decisions





