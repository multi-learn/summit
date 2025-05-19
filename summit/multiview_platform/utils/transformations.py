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
# * Sokol Koço <sokol.koco_AT_lis-lab.fr>
# * Cécile Capponi <cecile.capponi_AT_univ-amu.fr>
# * Dominique Benielli <dominique.benielli_AT_univ-amu.fr>
# * Baptiste Bauvin <baptiste.bauvin_AT_univ-amu.fr>
#
# Description:
# -----------
#
#
#
# Version:
# -------
#
# * multiview_generator version = 0.0.1
#
# Licence:
# -------
#
# License: New BSD License
#
#
# ######### COPYRIGHT #########
#
import numpy as np


def sign_labels(labels):
    """
    Returns a label array with (-1,1) as labels.
    If labels was already made of (-1,1), returns labels.
    If labels is made of (0,1), returns labels with all
    zeros transformed in -1.

    Parameters
    ----------
    labels

    The original label numpy array

    Returns
    -------
    A np.array with labels made of (-1,1)
    """
    if 0 in labels:
        return np.array([label if label != 0 else -1 for label in labels])
    else:
        return labels


def unsign_labels(labels):
    """
    The inverse function

    Parameters
    ----------
    labels

    Returns
    -------

    """
    if len(labels.shape) == 2:
        labels = labels.reshape((labels.shape[0],))
    if -1 in labels:
        return np.array([label if label != -1 else 0 for label in labels])
    else:
        return labels
