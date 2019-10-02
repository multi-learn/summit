import numpy as np

from multiview_platform.mono_multi_view_classifiers.multiview_classifiers.additions.diversity_utils import GlobalDiversityFusionClassifier


classifier_class_name = "EntropyFusion"


class EntropyFusion(GlobalDiversityFusionClassifier):

    def diversity_measure(self, classifiers_decisions, combination, y):
        _, nb_view, nb_examples = classifiers_decisions.shape
        scores = np.zeros((nb_view, nb_examples), dtype=int)
        for view_index, classifier_index in enumerate(combination):
            scores[view_index] = np.logical_not(
                np.logical_xor(classifiers_decisions[classifier_index, view_index],
                               y)
            )
        entropy_scores = np.sum(scores, axis=0)
        nb_view_matrix = np.zeros((nb_examples),
                                dtype=int) + nb_view - entropy_scores
        entropy_score = np.mean(np.minimum(entropy_scores, nb_view_matrix).astype(float) / (
                        nb_view - int(nb_view / 2)))
        return entropy_score
