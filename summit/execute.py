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
"""This is the execution module, used to execute the code"""

import os


def execute(config_path=None):  # pragma: no cover
    import sys

    from summit.multiview_platform import exec_classif
    if config_path is None:
        exec_classif.exec_classif(sys.argv[1:])
    else:
        if config_path == "example 0":
            config_path = os.path.join(
                os.path.dirname(
                    os.path.realpath(__file__)),
                "examples",
                "config_files",
                "config_example_0.yml")
        elif config_path == "example 1":
            config_path = os.path.join(
                os.path.dirname(
                    os.path.realpath(__file__)),
                "examples",
                "config_files",
                "config_example_1.yml")
        elif config_path == "example 2.1.1":
            config_path = os.path.join(
                os.path.dirname(
                    os.path.realpath(__file__)),
                "examples",
                "config_files",
                "config_example_2_1_1.yml")
        elif config_path == "example 2.1.2":
            config_path = os.path.join(
                os.path.dirname(
                    os.path.realpath(__file__)),
                "examples",
                "config_files",
                "config_example_2_1_2.yml")
        elif config_path == "example 2.2":
            config_path = os.path.join(
                os.path.dirname(
                    os.path.realpath(__file__)),
                "examples",
                "config_files",
                "config_example_2_2.yml")
        elif config_path == "example 2.3":
            config_path = os.path.join(
                os.path.dirname(
                    os.path.realpath(__file__)),
                "examples",
                "config_files",
                "config_example_2_3.yml")
        elif config_path == "example 3":
            config_path = os.path.join(
                os.path.dirname(
                    os.path.realpath(__file__)),
                "examples",
                "config_files",
                "config_example_3.yml")
        exec_classif.exec_classif(["--config_path", config_path])


if __name__ == "__main__":
    execute()
