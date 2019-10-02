import numpy as np

from multiview_platform.mono_multi_view_classifiers.multiview_classifiers.additions.diversity_utils import GlobalDiversityFusion


classifier_class_name = "DifficultyFusion"


class DifficultyFusion(GlobalDiversityFusion):

    def diversity_measure(self, classifiers_decisions, combination, y):

        _, nb_view, nb_examples = classifiers_decisions.shape
        scores = np.zeros((nb_view, nb_examples), dtype=int)
        for view_index, classifier_index in enumerate(combination):
            scores[view_index] = np.logical_not(
                    np.logical_xor(classifiers_decisions[classifier_index,
                                                         view_index],
                                   y)
                )
        # difficulty_scores = np.sum(scores, axis=0)
        # TODO : Check computing method
        difficulty_score = np.mean(
            np.var(
                np.array([
                             np.sum((scores==view_index), axis=1)/float(nb_view)
                             for view_index in range(len(combination)+1)])
                , axis=0)
        )
        return difficulty_score




