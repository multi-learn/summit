import numpy as np

from multiview_platform.mono_multi_view_classifiers.multiview_classifiers.additions.diversity_utils import CoupleDiversityFusionClassifier


classifier_class_name = "disagree_fusion"


class DisagreeFusion(CoupleDiversityFusionClassifier):

    def diversity_measure(self, first_classifier_decision, second_classifier_decision, _):
        return np.logical_xor(first_classifier_decision, second_classifier_decision)
