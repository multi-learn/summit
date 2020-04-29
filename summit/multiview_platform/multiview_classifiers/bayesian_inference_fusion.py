import numpy as np

from ..multiview_classifiers.additions.late_fusion_utils import \
    LateFusionClassifier
from ..utils.dataset import get_examples_views_indices

classifier_class_name = "BayesianInferenceClassifier"


class BayesianInferenceClassifier(LateFusionClassifier):
    def __init__(self, random_state, classifiers_names=None,
                 classifier_configs=None, nb_cores=1, weights=None,
                 rs=None):
        self.need_probas = True
        LateFusionClassifier.__init__(self, random_state=random_state,
                                      classifiers_names=classifiers_names,
                                      classifier_configs=classifier_configs,
                                      nb_cores=nb_cores,
                                      weights=weights,
                                      rs=rs)

    def predict(self, X, example_indices=None, view_indices=None):
        example_indices, view_indices = get_examples_views_indices(X,
                                                                   example_indices,
                                                                   view_indices)
        self._check_views(view_indices)
        if sum(self.weights) != 1.0:
            self.weights = self.weights / sum(self.weights)

        view_scores = []
        for index, view_index in enumerate(view_indices):
            view_scores.append(np.power(
                self.monoview_estimators[index].predict_proba(
                    X.get_v(view_index,
                            example_indices)),
                self.weights[index]))
        view_scores = np.array(view_scores)
        predicted_labels = np.argmax(np.prod(view_scores, axis=0), axis=1)
        return predicted_labels
