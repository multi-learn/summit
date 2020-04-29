import numpy as np

from summit.multiview_platform.multiview_classifiers.additions.diversity_utils import \
    CoupleDiversityFusionClassifier

classifier_class_name = "DisagreeFusion"


class DisagreeFusion(CoupleDiversityFusionClassifier):

    def diversity_measure(self, first_classifier_decision,
                          second_classifier_decision, _):
        return np.logical_xor(first_classifier_decision,
                              second_classifier_decision)
