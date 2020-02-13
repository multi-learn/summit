import numpy as np

from ..multiview_classifiers.additions.late_fusion_utils import LateFusionClassifier
from ..utils.dataset import get_examples_views_indices

classifier_class_name =  "MajorityVoting"

class VotingIndecision(Exception):
    pass

class MajorityVoting(LateFusionClassifier):
    def __init__(self, random_state, classifiers_names=None,
                 classifier_configs=None, weights=None, nb_cores=1, rs=None):
        self.need_probas=False
        super(MajorityVoting, self).__init__(random_state=random_state,
                                      classifiers_names=classifiers_names,
                                      classifier_configs=classifier_configs,
                                      nb_cores=nb_cores,
                                      weights=weights,
                                      rs=rs)

    def predict(self, X, example_indices=None, view_indices=None):
        examples_indices, views_indices = get_examples_views_indices(X,
                                                                     example_indices,
                                                                     view_indices)

        n_examples = len(examples_indices)
        votes = np.zeros((n_examples, X.get_nb_class(example_indices)), dtype=float)
        monoview_decisions = np.zeros((len(examples_indices), X.nb_view), dtype=int)
        for index, view_index in enumerate(views_indices):
            monoview_decisions[:, index] = self.monoview_estimators[index].predict(
                X.get_v(view_index, examples_indices))
        for example_index in range(n_examples):
            for view_index, feature_classification in enumerate(monoview_decisions[example_index, :]):
                votes[example_index, feature_classification] += self.weights[view_index]
            nb_maximum = len(np.where(votes[example_index] == max(votes[example_index]))[0])
            if nb_maximum == X.nb_view:
                raise VotingIndecision("Majority voting can't decide, each classifier has voted for a different class")

        predicted_labels = np.argmax(votes, axis=1)
        # Can be upgraded by restarting a new classification process if
        # there are multiple maximums ?:
        # 	while nbMaximum>1:
        # 		relearn with only the classes that have a maximum number of vote
        return predicted_labels
