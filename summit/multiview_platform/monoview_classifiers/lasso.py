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
from sklearn.linear_model import Lasso as LassoSK

from ..monoview.monoview_utils import BaseMonoviewClassifier
from summit.multiview_platform.utils.hyper_parameter_search import CustomUniform, CustomRandint

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "Lasso"


class Lasso(LassoSK, BaseMonoviewClassifier):
    """
     This class is an adaptation of scikit-learn's `Lasso <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>`_


     """

    def __init__(self, random_state=None, alpha=1.0,
                 max_iter=10, warm_start=False, **kwargs):
        LassoSK.__init__(self,
                         alpha=alpha,
                         max_iter=max_iter,
                         warm_start=warm_start,
                         random_state=random_state
                         )
        self.param_names = ["max_iter", "alpha", "random_state"]
        self.classed_params = []
        self.distribs = [CustomRandint(low=1, high=300),
                         CustomUniform(), [random_state]]
        self.weird_strings = {}

    def fit(self, X, y, check_input=True):
        neg_y = np.copy(y)
        neg_y[np.where(neg_y == 0)] = -1
        LassoSK.fit(self, X, neg_y)
        # self.feature_importances_ = self.coef_/np.sum(self.coef_)
        return self

    def predict(self, X):
        prediction = LassoSK.predict(self, X)
        signed = np.sign(prediction)
        signed[np.where(signed == -1)] = 0
        return signed
