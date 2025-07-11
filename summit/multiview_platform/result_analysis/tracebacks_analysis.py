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


def publish_tracebacks(directory, database_name, labels_names, tracebacks,
                       iter_index):
    if tracebacks:
        with open(os.path.join(directory, database_name +
                               "-iter" + str(iter_index) +
                               "-tacebacks.txt"),
                  "w") as traceback_file:
            failed_list = save_dict_to_text(tracebacks, traceback_file)
        flagged_list = [_ + "-iter" + str(iter_index) for _ in failed_list]
    else:
        flagged_list = {}
    return flagged_list


def save_dict_to_text(dictionnary, output_file):
    # TODO : smarter way must exist
    output_file.write("Failed algorithms : \n\t" + ",\n\t".join(
        dictionnary.keys()) + ".\n\n\n")
    for key, value in dictionnary.items():
        output_file.write(key)
        output_file.write("\n\n")
        output_file.write(value)
        output_file.write("\n\n\n")
    return dictionnary.keys()


def save_failed(failed_list, directory):
    with open(os.path.join(directory, "failed_algorithms.txt"),
              "w") as failed_file:
        failed_file.write(
            "The following algorithms sent an error, the tracebacks are stored "
            "in the coressponding directory :\n")
        failed_file.write(", \n".join(failed_list) + ".")
