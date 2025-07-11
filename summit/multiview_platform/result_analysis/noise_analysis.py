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
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# from matplotlib.patches import Patch
#
#
# def plot_results_noise(directory, noise_results, metric_to_plot, name,
#                        width=0.1):
#     avail_colors = ["tab:blue", "tab:orange", "tab:brown", "tab:gray",
#                     "tab:olive", "tab:red", ]
#     colors = {}
#     lengend_patches = []
#     noise_levels = np.array([noise_level for noise_level, _ in noise_results])
#     df = pd.DataFrame(
#         columns=['noise_level', 'classifier_name', 'mean_score', 'score_std'], )
#     if len(noise_results) > 1:
#         width = np.min(np.diff(noise_levels))
#     for noise_level, noise_result in noise_results:
#         classifiers_names, meaned_metrics, metric_stds = [], [], []
#         for noise_result in noise_result:
#             classifier_name = noise_result[0].split("-")[0]
#             if noise_result[1] is metric_to_plot:
#                 classifiers_names.append(classifier_name)
#                 meaned_metrics.append(noise_result[2])
#                 metric_stds.append(noise_result[3])
#                 if classifier_name not in colors:
#                     try:
#                         colors[classifier_name] = avail_colors.pop(0)
#                     except IndexError:
#                         colors[classifier_name] = "k"
#         classifiers_names, meaned_metrics, metric_stds = np.array(
#             classifiers_names), np.array(meaned_metrics), np.array(metric_stds)
#         sorted_indices = np.argsort(-meaned_metrics)
#         for index in sorted_indices:
#             row = pd.DataFrame(
#                 {'noise_level': noise_level,
#                  'classifier_name': classifiers_names[index],
#                  'mean_score': meaned_metrics[index],
#                  'score_std': metric_stds[index]}, index=[0])
#             df = pd.concat([df, row])
#             plt.bar(noise_level, meaned_metrics[index], yerr=metric_stds[index],
#                     width=0.5 * width, label=classifiers_names[index],
#                     color=colors[classifiers_names[index]])
#     for classifier_name, color in colors.items():
#         lengend_patches.append(Patch(facecolor=color, label=classifier_name))
#     plt.legend(handles=lengend_patches, loc='lower center',
#                bbox_to_anchor=(0.5, 1.05), ncol=2)
#     plt.ylabel(metric_to_plot)
#     plt.title(name)
#     plt.xticks(noise_levels)
#     plt.xlabel("Noise level")
#     plt.savefig(os.path.join(directory, name + "_noise_analysis.png"))
#     plt.close()
#     df.to_csv(os.path.join(directory, name + "_noise_analysis.csv"))
