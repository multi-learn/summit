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

from summit.multiview_platform.result_analysis import duration_analysis


class FakeClassifierResult:

    def __init__(self, i=0):
        self.i = i
        if i == 0:
            self.hps_duration = 10
            self.fit_duration = 12
            self.pred_duration = 15
        else:
            self.hps_duration = 1
            self.fit_duration = 2
            self.pred_duration = 5

    def get_classifier_name(self):
        if self.i == 0:
            return 'test1'
        else:
            return 'test2'


class Test_get_duration(unittest.TestCase):

    def test_simple(self):
        results = [FakeClassifierResult(), FakeClassifierResult(i=1)]
        durs = duration_analysis.get_duration(results)
        pd.testing.assert_frame_equal(durs,
                                      pd.DataFrame(index=['test1', 'test2'],
                                                   columns=[
                                                       'hps', 'fit', 'pred'],
                                                   data=np.array([np.array([10, 12, 15]),
                                                                  np.array([1, 2, 5])]),
                                                   dtype=object))
