====================================================
Example 3 : Understanding the statistical iterations
====================================================

Context
-------

In the previous example, we have seen that in order to output meaningful results, the platform splits the input dataset in a training set and a testing set.
However, even if the split is done at random, one can draw a lucky (or unlucky) split and have great (or poor) performance on this specific split.
To settle this issue, the platform can run on multiple splits and return the mean.


How to use it
-------------

This feature is controlled by a single argument : ``stats_iter:`` in the config file.
Modifying this argument and setting more than one ``stats_iter`` will slightly modify the result directory's structure.
Indeed, as the platform will perform a benchmark on multiple train/test split, the result directory will be larger in order to keep all the individual results.
In terms of pseudo-code, if one uses HPO, it adds a for loop on the pseudo code displayed in example 2 ::


    for each statistical iteration :
        ┌
        |for each monoview classifier:
        |    for each view:
        |        for each draw:
        |            for each fold:
        |                learn the classifier on all-1 folds and test it on 1
        |            get the mean performance
        |        get the best hyper-parameter set
        |        learn on the whole training set
        |and
        |for each multiview classifier:
        |    for each draw:
        |        for each fold:
        |            learn the classifier on all-1 folds and test it on 1
        |        get the mean performance
        |    get the best hyper-parameter set
        |    learn on the whole training set
        └

The result directory will be structured as :

.. code-block:: bash

    | started_1560_12_25-15_42
    | ├── iter_1
    | | ├── train_labels.csv
    | | └── train_indices.csv
    | | ├── 1560_12_25-15_42-*-LOG.log
    | | ├── config_file.yml
    | | ├── *-accuracy_score.
    | | ├── *-accuracy_score-class.html
    | | ├── *-accuracy_score.html
    | | ├── *-accuracy_score.csv
    | | ├── *-f1_score.png
    | | ├── *-f1_score.csv
    | | ├── *-f1_score-class.html
    | | ├── *-f1_score.html
    | | ├── *-error_analysis_2D.png
    | | ├── *-error_analysis_2D.html
    | | ├── *-error_analysis_bar.png
    | | ├── *-error_analysis_bar.HTML
    | | ├── *-bar_plot_data.csv
    | | ├── *-2D_plot_data.csv
    | | ├── feature_importances
    | | ├── [..
    | | ├── ..]
    | | ├── adaboost
    | | |   ├── ViewNumber0
    | | |   |   ├── *-summary.txt
    | | |   |   ├── <other classifier dependant files>
    | | |   ├── ViewNumber1
    | | |   |   ├── *-summary.txt
    | | |   |   ├── <other classifier dependant files>
    | | |   ├── ViewNumber2
    | | |   |   ├── *-summary.txt
    | | |   |   ├── <other classifier dependant files>
    | | ├── decision_tree
    | | |   ├── ViewNumber0
    | | |   |  ├── <summary & classifier dependant files>
    | | |   ├── ViewNumber1
    | | |   |  ├── <summary & classifier dependant files>
    | | |   ├── ViewNumber2
    | | |   |  ├── <summary & classifier dependant files>
    | | ├── [..
    | | ├── ..]
    | | ├── weighted_linear_late_fusion
    | | |   ├── <summary & classifier dependant files>
    | | ├── [..
    | | ├── ..]
    | ├── iter_2
    | | ├── [..
    | | ├── ..]
    | ├── [..
    | ├── ..]
    | ├── train_labels.csv
    | └── train_indices.csv
    | ├── 1560_12_25-15_42-*-LOG.log
    | ├── config_file.yml
    | ├── *-accuracy_score.png
    | ├── *-accuracy_score.csv
    | ├── *-accuracy_score.html
    | ├── *-accuracy_score-class.html
    | ├── *-f1_score.png
    | ├── *-f1_score.csv
    | ├── *-f1_score.html
    | ├── *-f1_score-class.html
    | ├── *-error_analysis_2D.png
    | ├── *-error_analysis_2D.html
    | ├── *-error_analysis_bar.png
    | ├── *-error_analysis_bar.html
    | ├── *-bar_plot_data.csv
    | ├── *-2D_plot_data.csv
    | ├── feature_importances
    | | ├── *-ViewNumber0-feature_importance.html
    | | ├── *-ViewNumber0-feature_importance_dataframe.csv
    | | ├── *-ViewNumber1-feature_importance.html
    | | ├── *-ViewNumber1-feature_importance_dataframe.csv
    | | ├── *-ViewNumber2-feature_importance.html
    | | ├── *-ViewNumber2-feature_importance_dataframe.csv
    | └── random_state.pickle

If you look closely, nearly all the files from Example 1 are in each ``iter_`` directory, and some files have appeared, in which the main figures are saved.
So, the files stored in ``started_1560_12_25-15_42/`` are the one that show the mean results on all the statistical iterations.
For example, ``started_1560_12_25-15_42/*-accuracy_score.png`` looks like :

.. raw:: html
    ./images/accuracy_mean.html

    The main difference between this plot an the one from Example 1 is that here, the scores are means over all the statistical iterations, and the standard deviations are plotted as vertical lines on top of the bars and printed after each score under the bars as "± <std>".

Then, each iteration's directory regroups all the results, structured as in Example 1.



Example
<<<<<<<


Duration
<<<<<<<<

Increasing the number of statistical iterations can be costly in terms of computational resources


