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

from multimodal.kernels.lpMKL import MKL

from .additions.kernel_learning import KernelClassifier, KernelConfigGenerator
from ..utils.hyper_parameter_search import CustomUniform

classifier_class_name = "LPNormMKL"

class LPNormMKL(KernelClassifier, MKL):
    def __init__(self, random_state=None, lmbda=0.1, nystrom_param=1, n_loops=50,
                 precision=0.0001, use_approx=True, kernel="rbf",
                 kernel_params=None):
        KernelClassifier.__init__(self, random_state)
        MKL.__init__(self, lmbda, nystrom_param=nystrom_param,
                                                      kernel=kernel,
                                                      n_loops=n_loops,
                                                      precision=precision,
                                                      use_approx=use_approx,
                                                      kernel_params=kernel_params)
        self.param_names = ["lmbda", "kernel", "kernel_params"]
        self.distribs = [CustomUniform(), ['rbf', 'additive_chi2', 'poly' ],
                         KernelConfigGenerator()]

    def fit(self, X, y, train_indices=None, view_indices=None):
        formatted_X, train_indices = self.format_X(X, train_indices, view_indices)
        self.init_kernels(nb_view=len(formatted_X))


        return MKL.fit(self, formatted_X, y[train_indices])

    def predict(self, X, sample_indices=None, view_indices=None):
        new_X, _ = self.format_X(X, sample_indices, view_indices)
        return self.extract_labels(MKL.predict(self, new_X))



