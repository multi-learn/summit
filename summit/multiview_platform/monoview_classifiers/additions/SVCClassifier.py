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
from sklearn.svm import SVC


class SVCClassifier(SVC):

    def __init__(self, random_state=None, kernel='rbf', C=1.0, degree=3,
                 **kwargs):
        super(SVCClassifier, self).__init__(
            C=C,
            kernel=kernel,
            degree=degree,
            probability=True,
            max_iter=1000,
            random_state=random_state
        )
        self.classed_params = []
        self.weird_strings = {}
