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
import inspect

from ...multiview.multiview_utils import get_monoview_classifier
from ...utils.multiclass import get_mc_estim


class BaseFusionClassifier():

    def init_monoview_estimator(self, classifier_name, classifier_config,
                                classifier_index=None, multiclass=False):
        if classifier_index is not None:
            if classifier_config is not None:
                classifier_configs = classifier_config
            else:
                classifier_configs = None
        else:
            classifier_configs = classifier_config
        if classifier_configs is not None and classifier_name in classifier_configs:
            if 'random_state' in inspect.getfullargspec(
                    get_monoview_classifier(classifier_name).__init__).args:
                estimator = get_monoview_classifier(classifier_name)(
                    random_state=self.random_state,
                    **classifier_configs[classifier_name])
            else:
                estimator = get_monoview_classifier(classifier_name)(
                    **classifier_configs[classifier_name])
        else:
            if 'random_state' in inspect.getfullargspec(
                    get_monoview_classifier(classifier_name).__init__).args:
                estimator = get_monoview_classifier(classifier_name)(
                    random_state=self.random_state)
            else:
                estimator = get_monoview_classifier(classifier_name)()

        return get_mc_estim(estimator, random_state=self.random_state,
                            multiview=False, multiclass=multiclass)
