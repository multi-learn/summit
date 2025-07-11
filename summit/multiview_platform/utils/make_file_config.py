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
import importlib
import inspect


class ConfigurationMaker():
    """
    Find the name of the classifier from the dict classier to report



    """
    _path_classifier_mono = 'summit/mono_multi_view_classifier/monoview_classifiers'
    _path_classifier_multi = 'summit/mono_multi_view_classifier/multiview_classifier'

    def __init__(self, classifier_dict=None):
        if classifier_dict is None:
            classifier_dict = {"0": ['mono', 'Adaboost',
                                     'summit.multiview_platform.monoview_classifiers.adaboost']}
        names = []
        for key, val in classifier_dict.items():
            mymodule = importlib.import_module(val[2])
            names.append(self._get_module_name(mymodule))
            monInstance = getattr(mymodule, val[1])

    def _get_module_name(self, mymodule):
        for name in dir(mymodule):
            att = getattr(mymodule, name)
            try:
                getattr(att, "__module__")
                if att.__module__.startswith(mymodule.__name__):
                    if inspect.isclass(att):
                        if att == val[1]:
                            return name
            except Exception:
                return None
        return None


if __name__ == '__main__':
    ConfigurationMaker()
