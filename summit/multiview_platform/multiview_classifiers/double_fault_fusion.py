import numpy as np

from summit.multiview_platform.multiview_classifiers.additions.diversity_utils import \
    CoupleDiversityFusionClassifier

classifier_class_name = "DoubleFaultFusion"


class DoubleFaultFusion(CoupleDiversityFusionClassifier):

    """
    This classifier is inspired by
    Kuncheva, Ludmila & Whitaker, Chris. (2000). Measures of Diversity in
    Classifier Ensembles.
    It find the subset of monoview classifiers with the best double fault
    """

    def diversity_measure(self, first_classifier_decision,
                          second_classifier_decision, y):
        return np.logical_and(np.logical_xor(first_classifier_decision, y),
                              np.logical_xor(second_classifier_decision, y))
