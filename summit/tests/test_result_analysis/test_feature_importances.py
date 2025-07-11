# -*- coding: utf-8 -*-
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
# * Baptiste Bauvin <baptiste.bauvin_AT_univ-amu.fr>
# * Sokol Koço <sokol.koco_AT_lis-lab.fr>
# * Cécile Capponi <cecile.capponi_AT_univ-amu.fr>
# * Dominique Benielli <dominique.benielli_AT_univ-amu.fr>
#
#
# Description:
# -----------
#
# Supervised MultiModal Integration Tool's Readme
# This project aims to be an easy-to-use solution to run a prior benchmark on a dataset and evaluate mono- & multi-view algorithms capacity to classify it correctly.
#
# Version:
# -------
#
# * summit-multi-learn version = 0.0.2
#
# Licence:
# -------
#
# License: New BSD License : BSD-3-Clause
#
#
# ######### COPYRIGHT #########
#
#
#
#
import unittest
import numpy as np
import pandas as pd

from summit.multiview_platform.result_analysis import feature_importances
from summit.multiview_platform.monoview.monoview_utils import MonoviewResult


class FakeClassifier:
    def __init__(self, i=0):
        self.feature_importances_ = [i, i + 1]
        self.view_index=0


class FakeClassifierResult(MonoviewResult):

    def __init__(self, i=0):
        self.i = i
        self.hps_duration = i * 10
        self.fit_duration = (i + 2) * 10
        self.pred_duration = (i + 5) * 10
        self.clf = FakeClassifier(i)
        self.view_name = 'testview' + str(i)
        self.classifier_name = "test" + str(i)
        self.view_index = 0

    def get_classifier_name(self):
        return self.classifier_name


class Test_get_duration(unittest.TestCase):

    def test_simple(self):
        results = [FakeClassifierResult(), FakeClassifierResult(i=1)]
        feat_importance = feature_importances.get_feature_importances(results, feature_ids=[["a", "b"]], view_names=["v"])
        pd.testing.assert_frame_equal(feat_importance["testview1"],
                                      pd.DataFrame(index=["a", "b"], columns=['test1'],
                                                   data=np.array(
                                                       [1, 2]).reshape((2, 1)),
                                                   ), check_dtype=False)
