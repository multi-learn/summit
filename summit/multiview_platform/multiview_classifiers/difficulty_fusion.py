import numpy as np

from summit.multiview_platform.multiview_classifiers.additions.diversity_utils import \
    GlobalDiversityFusionClassifier

classifier_class_name = "DifficultyFusion"


class DifficultyFusion(GlobalDiversityFusionClassifier):

    """
    This classifier is inspired by Kuncheva, Ludmila & Whitaker, Chris. (2000). Measures of Diversity in Classifier Ensembles.
    It find the subset of monoview classifiers with the best difficulty
    """

    def diversity_measure(self, classifiers_decisions, combination, y):
        _, nb_view, nb_samples = classifiers_decisions.shape
        scores = np.zeros((nb_view, nb_samples), dtype=int)
        for view_index, classifier_index in enumerate(combination):
            scores[view_index, :] = np.logical_not(
                np.logical_xor(classifiers_decisions[classifier_index,
                                                     view_index],
                               y)
            )
        # Table of the nuber of views that succeeded for each sample :
        difficulty_scores = np.sum(scores, axis=0)

        difficulty_score = np.var(
            np.array([
                np.sum((difficulty_scores == view_index))
                for view_index in range(len(combination) + 1)])
        )
        return difficulty_score
