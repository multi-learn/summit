
===============================================
Example 1 : First steps with Multiview Platform
===============================================

Context
--------------------


This platform aims at running multiple state-of-the-art classifiers on a multiview dataset in a classification context.
It has been developed in order to get a baseline on common algorithms for any classification task.

Adding a new classifier (monoview and/or multiview) as been made as simple as possible in order for users to be able to
customize the set of classifiers and test their performances in a controlled environment.




Introduction to this tutorial
-----------------------------

This tutorial will show you how to use the platform on simulated data, for the simplest problem : biclass classification.

The data is naively generated TODO : Keep the same generator ?


Getting started
---------------

**Importing the platform's execution function**

.. code-block:: python

   >>> from multiview_platform.execute import execute

**Understanding the config file**

The config file that will be used in this example is located in ``multiview-machine-learning-omis/multiview_platform/examples/config_files/config_exmaple_1.yml``

We will decrypt the main arguments :

+ The first part of the arguments are the basic ones :

    - ``log: True`` allows to print the log in the terminal,
    - ``name: ["plausible"]`` uses the plausible simulated dataset,
    - ``random_state: 42`` fixes the random state of this benchmark, it is useful for reproductibility,
    - ``full: True`` the benchmark will used the full dataset,
    - ``res_dir: "examples/results/example_1/"`` the results will be saved in ``multiview-machine-learning-omis/multiview_platform/examples/results/example_1``

+ Then the classification-related arguments :

    - ``split: 0.8`` means that 80% of the dataset will be used to test the different classifiers and 20% to train them
    - ``type: ["monoview", "multiview"]`` allows for monoview and multiview algorithms to be used in the benchmark
    - ``algos_monoview: ["all"]`` runs on all the available monoview algorithms (same for ``algos_muliview``)
    - ``metrics: ["accuracy_score", "f1_score"]`` means that the benchmark will evaluate the performance of each algortihms on these two metrics.

+ Then, the two following categories are algorithm-related and contain all the default values for the hyper-parameters.

**Start the benchmark**

During the whole benchmark, the log file will be printed in the terminal. To start the benchmark run :

.. code-block:: python

   >>> execute()

The execution should take less than five minutes. We will first analyze the results and parse through the information the platform output.


**Understanding the results**

The result structure can be startling at first, but as the platform provides a lot of information, it has to be organized.

The results are stored in ``multiview_platform/examples/results/example_1/``. Here, you will find a directory with the name of the database used for the benchmark, here : ``plausible/``
Then, a directory with the amount of noise in the experiments, we didn't add any, so ``n_0/`` finally, a directory with
the date and time of the beginning of the experiment. Let's say you started the benchmark on the 25th of December 1560,
at 03:42 PM, the directory's name should be ``started_1560_12_25-15_42/``.

From here the result directory has the structure that follows  :

.. code-block:: bash

    | started_1560_12_25-15_42
    | ├── adaboost
    | |   ├── ViewNumber0
    | |   |   ├── *-summary.txt
    | |   |   ├── <other classifier dependant files>
    | |   ├── ViewNumber1
    | |   |   ├── *-summary.txt
    | |   |   ├── <other classifier dependant files>
    | |   ├── ViewNumber2
    | |   |   ├── *-summary.txt
    | |   |   ├── <other classifier dependant files>
    | ├── decision_tree
    | |   ├── ViewNumber0
    | |   |  ├── <summary & classifier dependant files>
    | |   ├── ViewNumber1
    | |   |  ├── <summary & classifier dependant files>
    | |   ├── ViewNumber2
    | |   |  ├── <summary & classifier dependant files>
    | ├── [..
    | ├── ..]
    | ├── weighted_linear_late_fusion
    | |   ├── <summary & classifier dependant files>
    | ├── [..
    | ├── ..]
    | ├── train_labels.csv
    | └── train_indices.csv
    | ├── 1560_12_25-15_42-*-LOG.log
    | ├── config_file.yml
    | ├── *-accuracy_score.png
    | ├── *-accuracy_score.csv
    | ├── *-f1_score.png
    | ├── *-f1_score.csv
    | ├── *-error_analysis_2D.png
    | ├── *-error_analysis_2D.html
    | ├── *-error_analysis_bar.png
    | ├── feature_importances
    | | ├── *-ViewNumber0-feature_importance.html
    | | ├── *-ViewNumber0-feature_importance_dataframe.csv
    | | ├── *-ViewNumber1-feature_importance.html
    | | ├── *-ViewNumber1-feature_importance_dataframe.csv
    | | ├── *-ViewNumber2-feature_importance.html
    | | ├── *-ViewNumber2-feature_importance_dataframe.csv
    | ├── *-bar_plot_data.csv
    | ├── *-2D_plot_data.csv
    | └── random_state.pickle


The structure can seem complex, but it priovides a lot of information, from the most general to the most precise.

Let's comment each file :

``*-accuracy_score.png`` and ``*-accuracy_score.csv``
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

These files contain the scores of each classifier for the accuracy metric, ordered with le best ones on the right and
the worst ones on the left, as an image or as as csv matrix.
The image version is as follows :

.. figure:: ./images/accuracy.png
    :scale: 25

    This is a bar plot showing the score on the training set (light gray), and testing set (dark gray). For each
    monoview classifier, on each view and or each multiview classifier, the scores are printed over the name, under each bar.
    It is highly recommended to click on the image to be able to zoom.

The csv file is a matrix with the score on train stored in the first row and the score on test stored in the second one. Each classifier is presented in a row. It is loadable with pandas.

Similar files have been generated for the f1 metric (``*-f1_score.png`` and ``*-f1_score.csv``).


``*-error_analysis_2D.png`` and ``*-error_analysis_2D.html``
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

In these files, one can visualize the success or failure of each classifier on each example.

Below, ``*-error_analysis_2D.html`` is displayed.

It is the representation of a matrix, where the rows are the examples, and the columns are the classifiers.

If a classifier (Lasso on the first view for example) missclassified an example (example number 75 for examples),
a black rectangle is printed in the row corresponding to example 75 and the column corresponding to Lasso-ViewNumber0,
and if the classifier successfully classified the example, a white rectangle is printed.

.. raw:: html
    :file: ./images/error_2D.html

This figure is the html version of the classifiers errors' visualization. It is interactive, so, by hovering it, the information on
each classifier and example is printed. The classifiers are ordered as follows:

From left to right : all the monoview classifiers on ViewNumber0, all the ones on ViewNumber1, ..., then at the far right, the multiview classifiers

This html image is also available in ``.png`` format, but is then not interactive, so harder to analyze.

In terms of information, this is useful to detect possible outlier examples in the dataset and failing classifers.

For example, a mainly black horizontal line for an example means that it has been missclassified by most of the classifiers.
It could mean that the example is incorrectly labeled in the dataset or is very hard to classify.

Symmetrically, a mainly-black column means that a classifier spectacularly failed on the asked task.

On the figure displayed here, each view is visible as most monoview classifiers fail on the same examples inside the view.
It is an understandable behaviour as the Plausible dataset's view are generated and noised independently.
Morever, as confirmed by the accuracy graph, four monoview classifiers classified all the example to the same class,
and then, display a black half-column.

The data used to generate those matrices is available in ``*-2D_plot_data.csv``

``*-error_analysis_bar.png``
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

This file is a different way to visualize the same information as the two previous ones. Indeed, it is a bar plot,
with a bar for each example, counting the number of classifiers that failed to classify this particular example.

.. figure:: ./images/bar_error.png
    :scale: 25

    The bar plot showing for each example how many classifiers failed on it.

The data used to generate this graph is available in ``*-bar_plot_data.csv``

``config_file.yml``
<<<<<<<<<<<<<<<<<<<

This is a copy of the configuration file used to run the experiment.

``random_state.pickle``
<<<<<<<<<<<<<<<<<<<<<<<

A save of the numpy random state that was used for the experiment, it is mainly useful if no seed is specified in the config file.

``1560_12_25-15_42-*-LOG.log``
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

The log file

Classifier-dependant files
<<<<<<<<<<<<<<<<<<<<<<<<<<

For each classifier, at least one file is generated, called ``*-summary.txt``.

.. include:: ./images/summary.txt
    :literal:

This regroups the useful information on the classifiers configuration and it's performance. An interpretation section is
available for classifiers that present some interpretation-related information (as feature importance).