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

This feature is controlled by a single argument : ``stats_iter:`` in the ``Classification`` section of the config file.
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
    | ├── No-vs-Yes
    | | ├── 1560_12_25-15_42-*-accuracy_score.png
    | | ├── 1560_12_25-15_42-*-accuracy_score.csv
    | | ├── 1560_12_25-15_42-*-f1_score.png
    | | ├── 1560_12_25-15_42-*-f1_score.csv
    | | ├── 1560_12_25-15_42-*-error_analysis_2D.png
    | | ├── 1560_12_25-15_42-*-error_analysis_2D.html
    | | ├── 1560_12_25-15_42-*-error_analysis_bar.png
    | | ├── 1560_12_25-15_42-*-ViewNumber0-feature_importance.html
    | | ├── 1560_12_25-15_42-*-ViewNumber0-feature_importance_dataframe.csv
    | | ├── 1560_12_25-15_42-*-ViewNumber1-feature_importance.html
    | | ├── 1560_12_25-15_42-*-ViewNumber1-feature_importance_dataframe.csv
    | | ├── 1560_12_25-15_42-*-ViewNumber2-feature_importance.html
    | | ├── 1560_12_25-15_42-*-ViewNumber2-feature_importance_dataframe.csv
    | | ├── 1560_12_25-15_42-*-bar_plot_data.csv
    | | ├── 1560_12_25-15_42-*-2D_plot_data.csv
    | ├── iter_1
    | | ├── No-vs-Yes
    | | | ├── adaboost
    | | | |   ├── ViewNumber0
    | | | |   |   ├── 1560_12_25-15_42-*-summary.txt
    | | | |   |   ├── <other classifier dependant files>
    | | | |   ├── ViewNumber1
    | | | |   |   ├── 1560_12_25-15_42-*-summary.txt
    | | | |   |   ├── <other classifier dependant files>
    | | | |   ├── ViewNumber2
    | | | |   |   ├── 1560_12_25-15_42-*-summary.txt
    | | | |   |   ├── <other classifier dependant files>
    | | | ├── decision_tree
    | | | |   ├── ViewNumber0
    | | | |   |  ├── <summary & classifier dependant files>
    | | | |   ├── ViewNumber1
    | | | |   |  ├── <summary & classifier dependant files>
    | | | |   ├── ViewNumber2
    | | | |   |  ├── <summary & classifier dependant files>
    | | | ├── [..
    | | | ├── ..]
    | | | ├── weighted_linear_late_fusion
    | | | |   ├── <summary & classifier dependant files>
    | | | | ├── [..
    | | | | ├── ..]
    | | | ├── train_labels.csv
    | | │ └── train_indices.csv
    | ├── 1560_12_25-15_42-*-LOG.log
    | ├── config_file.yml
    | | ├── 1560_12_25-15_42-*-accuracy_score.png
    | | ├── 1560_12_25-15_42-*-accuracy_score.csv
    | | ├── 1560_12_25-15_42-*-f1_score.png
    | | ├── 1560_12_25-15_42-*-f1_score.csv
    | | ├── 1560_12_25-15_42-*-error_analysis_2D.png
    | | ├── 1560_12_25-15_42-*-error_analysis_2D.html
    | | ├── 1560_12_25-15_42-*-error_analysis_bar.png
    | | ├── 1560_12_25-15_42-*-ViewNumber0-feature_importance.html
    | | ├── 1560_12_25-15_42-*-ViewNumber0-feature_importance_dataframe.csv
    | | ├── 1560_12_25-15_42-*-ViewNumber1-feature_importance.html
    | | ├── 1560_12_25-15_42-*-ViewNumber1-feature_importance_dataframe.csv
    | | ├── 1560_12_25-15_42-*-ViewNumber2-feature_importance.html
    | | ├── 1560_12_25-15_42-*-ViewNumber2-feature_importance_dataframe.csv
    | | ├── 1560_12_25-15_42-*-bar_plot_data.csv
    | | ├── 1560_12_25-15_42-*-2D_plot_data.csv
    | ├── iter_2
    | | ├── No-vs-Yes
    | | | ├─[...
    | | | ├─...]
    | ├── iter_3
    | ├── [...
    | ├── ...]
    | └── random_state.pickle

If you look closely, nearly all the files from Example 1 are in each ``iter_`` directory, and a new ``No-vs-Yes`` directory ha appeared, in which the main figures are saved.
So, the files saved in ``started_1560_12_25-15_42/No-vs-Yes/`` are the one that show th mean results on all the statistical iterations.
For example, ``started_1560_12_25-15_42/No-vs-Yes/1560_12_25-15_42-*-accuracy_score.png`` looks like :

.. figure:: ./images/accuracy_mean.png
    :scale: 25

    The main difference between this plot an the one from Example 1 is that here, the scores are means over all the satatisitcal iterations, and the standard deviations are plotted as vertical lines on top of the bars and printed after each score under the bars as "± <std>".

Then, each iteration's directory regroups all the results, structured as in Example 1.



**Example with stats iter**

**Duration ??**



