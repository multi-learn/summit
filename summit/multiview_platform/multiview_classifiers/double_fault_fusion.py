# ######### COPYRIGHT #########
#
# Copyright(c) 2025
# -----------------
#
#
# * Université d'Aix Marseille (AMU) -
# * Centre National de la Recherche Scientifique (CNRS) -
# * Université de Toulon (UTLN).
# * Copyright © 2019-2025 AMU, CNRS, UTLN
#
# Contributors:
# ------------
#
# * Sokol Koço <sokol.koco_AT_lis-lab.fr>
# * Cécile Capponi <cecile.capponi_AT_univ-amu.fr>
# * Dominique Benielli <dominique.benielli_AT_univ-amu.fr>
# * Baptiste Bauvin <baptiste.bauvin_AT_univ-amu.fr>
#
# Description:
# -----------
#
#
#
# Version:
# -------
#
# * multiview_generator version = 0.0.1
#
# Licence:
# -------
#
# License: New BSD License
#
#
# ######### COPYRIGHT #########
#
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
