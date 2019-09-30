# -*- coding: utf-8 -*-


from __future__ import print_function, division

import os


def get_dataset_path(filename):
    """Return the absolute path of a reference dataset for tests

    - Input parameter:

    :param str filename: File name of the file containing reference data
        for tests (which must be in ``skgilearn/tests/datasets/``)

    - Output parameters:

    :returns: The absolute path where the file with name **filename** is stored
    :rtype: str

    """

    datasets_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(datasets_path, filename)
