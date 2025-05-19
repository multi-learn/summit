__version__ = "0.0.1"
"""
To be able to add another metric to the benchmark you must :

Create a .py file named after the metric
Define a score function
    Input :
        y_true : np array with the real labels
        y_pred : np array with the predicted labels
        kwargs : every argument that is specific to the metric
    Returns:
        score : the metric's score (float)
Define a get_scorer function
    Input :
        kwargs : every argument that is specific to the metric
    Returns :
        scorer : an object similar to an sk-learn scorer
Define a getConfig function
    Input :
        kwargs : every argument that is specific to the metric
    Output :
        config_string : A string that gives the name of the metric and explains how it is configured. Must end by
                        (lower is better) or (higher is better) to be able to analyze the preds
"""

import os

for module in os.listdir(os.path.dirname(os.path.realpath(__file__))):
    if module in ['__init__.py'] or module[-3:] != '.py':
        continue
    __import__(module[:-3], locals(), globals(), [], 1)
    pass
del os
