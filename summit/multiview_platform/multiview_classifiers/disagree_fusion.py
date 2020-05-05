import numpy as np

from summit.multiview_platform.multiview_classifiers.additions.diversity_utils import \
    CoupleDiversityFusionClassifier

classifier_class_name = "DisagreeFusion"


class DisagreeFusion(CoupleDiversityFusionClassifier):

    """
    This classifier is inspired by Kuncheva, Ludmila & Whitaker, Chris. (2000). Measures of Diversity in Classifier Ensembles.
    It find the subset of monoview classifiers with the best disagreement
    """

    def diversity_measure(self, first_classifier_decision,
                          second_classifier_decision, _):
        return np.logical_xor(first_classifier_decision,
                              second_classifier_decision)
