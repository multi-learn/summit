import os

for module in os.listdir(os.path.dirname(os.path.realpath(__file__))):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    __import__(module[:-3], locals(), globals(), [], 1)
del module
del os

"""
To be able to add a monoview Classifier to the benchmark, one has to :
Create a .py file named after the classifier
Define a canProbas function returning True or False whether the classifier is able to predict class probabilities
Define a fit function
    Input :
        DATASET : The data matrix used to fit the classifier
        CLASS_LABELS : The labels' array of the training set
        NB_CORES : The number of cores the classifier can use to train
        kwargs : Any argument specific to the classifier
    Output :
        classifier : A classifier object, similar to the sk-learn classifier object
Define a ***Search that search hyper parameters for the algorithm. Check HP optimization methods to get all the
different functions to provide (returning the parameters in the order of the kwargs dict for the fit function)
Define a getKWARGS function
    Input :
        KWARGSList : The list of all the arguments as written in the argument parser
    Output :
        KWARGSDict : a dictionnary of arguments matching the kwargs needed in train
Define a getConfig function that returns a string explaining the algorithm's config using a config dict or list
Add the arguments to configure the classifier in the parser in exec_classif.py
"""
