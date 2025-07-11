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
import os
import unittest

import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold

from summit.tests.utils import rm_tmp, tmp_path, test_dataset

from summit.multiview_platform.multiview import exec_multiview


class Test_init_constants(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()
        os.mkdir(tmp_path)

    @classmethod
    def tearDownClass(cls):
        rm_tmp()

    def test_simple(self):
        classifier_name, t_start, views_indices, \
            classifier_config, views, learning_rate, labels, output_file_name, \
            directory, base_file_name, metrics = exec_multiview.init_constants(
                kwargs={"view_names": ["ViewN0", "ViewN2", "ViewN1", ],
                        "view_indices": [0, 2, 1],
                        "classifier_name": "test_clf",
                        "test_clf": {}},
                classification_indices=[np.array([0, 1, 4, 2]), np.array([3])],
                metrics={"accuracy_score*": {}},
                name="test_dataset",
                nb_cores=1,
                k_folds=StratifiedKFold(n_splits=2),
                dataset_var=test_dataset,
                directory=tmp_path
            )
        self.assertEqual(classifier_name, "test_clf")
        self.assertEqual(views_indices, [0, 2, 1])
        self.assertEqual(classifier_config, {})
        self.assertEqual(views, ["ViewN0", "ViewN2", "ViewN1", ])
        self.assertEqual(learning_rate, 4 / 5)

    def test_exec_multiview_no_hps(self):
        res = exec_multiview.exec_multiview(
            directory=tmp_path,
            dataset_var=test_dataset,
            name="test_dataset",
            classification_indices=[np.array([0, 1, 4, 2]), np.array([3])],
            k_folds=StratifiedKFold(n_splits=2),
            nb_cores=1,
            database_type="", path="",
            labels_dictionary={0: "yes", 1: "no"},
            random_state=np.random.RandomState(42),
            labels=test_dataset.get_labels(),
            hps_method="None",
            hps_kwargs={},
            metrics=None,
            n_iter=30,
            **{"view_names": ["ViewN0", "ViewN2", "ViewN1", ],
               "view_indices": [0, 2, 1],
               "classifier_name": "weighted_linear_early_fusion",
               "weighted_linear_early_fusion": {}}
        )

    def test_exec_multiview(self):
        res = exec_multiview.exec_multiview(
            directory=tmp_path,
            dataset_var=test_dataset,
            name="test_dataset",
            classification_indices=[np.array([0, 1, 4, 2]), np.array([3])],
            k_folds=StratifiedKFold(n_splits=2),
            nb_cores=1,
            database_type="", path="",
            labels_dictionary={0: "yes", 1: "no"},
            random_state=np.random.RandomState(42),
            labels=test_dataset.get_labels(),
            hps_method="Grid",
            hps_kwargs={"param_grid":
                        {"monoview_classifier_config": [
                            {"max_depth": 3}, {"max_depth": 1}]},
                        },
            metrics=None,
            n_iter=30,
            **{"view_names": ["ViewN0", "ViewN2", "ViewN1", ],
               "view_indices": [0, 2, 1],
               "classifier_name": "weighted_linear_early_fusion",
               "weighted_linear_early_fusion": {}}
        )
