import numpy as np

from ..multiview_classifiers.additions.late_fusion_utils import \
    LateFusionClassifier
from ..utils.dataset import get_samples_views_indices

classifier_class_name = "MajorityVoting"


class VotingIndecision(Exception):
    pass


class MajorityVoting(LateFusionClassifier):

    """
    This classifier is a late fusion that builds a majority vote between the views
    """

    def __init__(self, random_state, classifiers_names=None,
                 classifier_configs=None, weights=None, nb_cores=1, rs=None):
        self.need_probas = False
        LateFusionClassifier.__init__(self, random_state=random_state,
                                      classifiers_names=classifiers_names,
                                      classifier_configs=classifier_configs,
                                      nb_cores=nb_cores,
                                      weights=weights,
                                      rs=rs)

    def predict(self, X, sample_indices=None, view_indices=None):
        samples_indices, view_indices = get_samples_views_indices(X,
                                                                  sample_indices,
                                                                  view_indices)
        self._check_views(view_indices)
        n_samples = len(samples_indices)
        votes = np.zeros((n_samples, X.get_nb_class(sample_indices)),
                         dtype=float)
        monoview_decisions = np.zeros((len(samples_indices), X.nb_view),
                                      dtype=int)
        for index, view_index in enumerate(view_indices):
            monoview_decisions[:, index] = self.monoview_estimators[
                index].predict(
                X.get_v(view_index, samples_indices))
        for sample_index in range(n_samples):
            for view_index, feature_classification in enumerate(
                    monoview_decisions[sample_index, :]):
                votes[sample_index, feature_classification] += self.weights[
                    view_index]
            nb_maximum = len(
                np.where(votes[sample_index] == max(votes[sample_index]))[0])
            if nb_maximum == X.nb_view:
                raise VotingIndecision(
                    "Majority voting can't decide, each classifier has voted for a different class")

        predicted_labels = np.argmax(votes, axis=1)
        # Can be upgraded by restarting a new classification process if
        # there are multiple maximums ?:
        # 	while nbMaximum>1:
        # 		relearn with only the classes that have a maximum number of vote
        return predicted_labels
