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
from sklearn.metrics import pairwise
import numpy as np

from ...multiview.multiview_utils import BaseMultiviewClassifier
from ...utils.hyper_parameter_search import CustomUniform, CustomRandint
from ...utils.transformations import sign_labels, unsign_labels
from ...utils.dataset import get_samples_views_indices

class KernelClassifier(BaseMultiviewClassifier):

    def __init__(self, random_state=None,):
        super().__init__(random_state)

    # def _compute_kernels(self, X, sample_indices, view_indices, ):
    #     new_X = {}
    #     for index, (kernel_function, kernel_config, view_index) in enumerate(
    #             zip(self.kernel_functions, self.kernel_configs, view_indices)):
    #         new_X[index] = kernel_function(X.get_v(view_index,
    #                                                sample_indices),
    #                                        **kernel_config)
    #     return new_X

    def format_X(self, X, sample_indices, view_indices):
        sample_indices, view_indices = get_samples_views_indices(X,
                                                                   sample_indices,
                                                                   view_indices)
        formatted_X = dict((index, X.get_v(view_index, sample_indices=sample_indices))
                     for index, view_index in enumerate(view_indices))

        return formatted_X, sample_indices

    def extract_labels(self, predicted_labels):
        signed_labels = np.sign(predicted_labels)
        return unsign_labels(signed_labels)

    def init_kernels(self, nb_view=2, ):
        if isinstance(self.kernel, KernelDistribution):
            self.kernel = self.kernel.draw(nb_view)
        elif isinstance(self.kernel, str):
            self.kernel = [self.kernel
                                     for _ in range(nb_view)]
        elif isinstance(self.kernel, list):
            pass

        if isinstance(self.kernel_params, KernelConfigDistribution):
            self.kernel_params = self.kernel_params.draw(nb_view)
            self.kernel_params = [kernel_config[kernel_name]
                                   for kernel_config, kernel_name
                                   in zip(self.kernel_params,
                                          self.kernel)]

        elif isinstance(self.kernel_params, dict):
            self.kernel_params = [self.kernel_params for _ in range(nb_view)]
        else:
            pass


class KernelConfigGenerator:

    def __init__(self):
        pass

    def rvs(self, random_state=None):
        return KernelConfigDistribution(seed=random_state.randint(1))


class KernelConfigDistribution:

    def __init__(self, seed=42):
        self.random_state=np.random.RandomState(seed)
        self.possible_config = {
            "additive_chi2": {"gamma": CustomUniform()},
            "rbf": {"gamma": CustomUniform()},
            "poly":{"degree": CustomRandint(1,4), "gamma":CustomUniform()}
        }

    def draw(self, nb_view):
        drawn_params = [{} for _ in range(nb_view)]
        for view_index in range(nb_view):
            for kernel_name, params_dict in self.possible_config.items():
                drawn_params[view_index][kernel_name] = {}
                for param_name, distrib in params_dict.items():
                    drawn_params[view_index][kernel_name][param_name] = distrib.rvs(self.random_state)
        return drawn_params


class KernelGenerator:

    def __init__(self):
        pass

    def rvs(self, random_state=None):
        return KernelDistribution(seed=random_state.randint(1))


class KernelDistribution:

    def __init__(self, seed=42):
        self.random_state=np.random.RandomState(seed)
        self.available_kernels = ["rbf"]

    def draw(self, nb_view):
        return list(self.random_state.choice(self.available_kernels, nb_view))
