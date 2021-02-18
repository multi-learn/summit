=======================================
Example 1 : First big step with |platf|
=======================================

Introduction to this tutorial
-----------------------------

This tutorial will show you how to use the platform on simulated data, for the simplest problem : vanilla multiclass classification.

The data is generated with a soon-to-be published multiview generator that allows to control redundancy, mutual error and complementarity among the views.

For all the following tutorials, we will use the same dataset.

A generated dataset to rule them all
------------------------------------

The :base_source:`dataset <summit/examples/data/doc_summit.hdf5>` that will be used in the samples consists in

+ 500 samples that are either
    + mis-described by all the views (labelled ``Mutual_error_*``),
    + well-described by all the views (labelled ``Redundant_*``),
    + well-described by the majority of the views (labelled ``Complementary_*``),
    + randomly well- or mis-described by the views (labelled ``sample_*``).

+ 8 balanced classes named ``'label_1'``, ..., ``'label_8'``,

+ 4 views named ``'generated_view_1'``, ..., ``'generated_view_4'``,
    + each view consisting in 3 features.

It has been parametrized with the following error matrix that encapsulates the quantity of information available in each view for each label :

+---------+--------+--------+--------+--------+
|         | View 1 | View 2 | View 3 | View 4 |
+=========+========+========+========+========+
| label_1 |  0.40  |  0.40  |  0.40  |  0.40  |
+---------+--------+--------+--------+--------+
| label_2 |  0.55  |  0.40  |  0.40  |  0.40  |
+---------+--------+--------+--------+--------+
| label_3 |  0.40  |  0.50  |  0.60  |  0.55  |
+---------+--------+--------+--------+--------+
| label_4 |  0.40  |  0.50  |  0.50  |  0.40  |
+---------+--------+--------+--------+--------+
| label_5 |  0.40  |  0.40  |  0.40  |  0.40  |
+---------+--------+--------+--------+--------+
| label_6 |  0.40  |  0.40  |  0.40  |  0.40  |
+---------+--------+--------+--------+--------+
| label_7 |  0.40  |  0.40  |  0.40  |  0.40  |
+---------+--------+--------+--------+--------+
| label_8 |  0.40  |  0.40  |  0.40  |  0.40  |
+---------+--------+--------+--------+--------+

So this means that the View 1 should make at least 40% error on label 1 and 65% on label 2.

Getting started
---------------

**Importing the platform's execution function**

.. code-block:: python

   >>> from summit.execute import execute

**Understanding the config file**

The config file that will be used in this example is available :base_source:`here <summit/examples/config_files/config_example_1.yml>`, let us decrypt the main arguments :

+ The first part regroups the basics :

    - :yaml:`log: True` (:base_source:`l4 <summit/examples/config_files/config_example_1.yml#L4>`) allows to print the log in the terminal,
    - :yaml:`name: ["summit_doc"]` (:base_source:`l6 <summit/examples/config_files/config_example_1.yml#L6>`) uses the plausible simulated dataset,
    - :yaml:`random_state: 42` (:base_source:`l18 <summit/examples/config_files/config_example_1.yml#L18>`) fixes the seed of the random state for this benchmark, it is useful for reproductibility,
    - :yaml:`full: True`  (:base_source:`l22 <summit/examples/config_files/config_example_1.yml#L22>`) means the benchmark will use the full dataset,
    - :yaml:`res_dir: "examples/results/example_1/"` (:base_source:`l26 <summit/examples/config_files/config_example_1.yml#L26>`) saves the results in ``summit/summit/examples/results/example_1``

+ Then the classification-related arguments :

    - :yaml:`split: 0.25` (:base_source:`l35 <summit/examples/config_files/config_example_1.yml#L35>`) means that 75% of the dataset will be used to test the different classifiers and 25% to train them,
    - :yaml:`type: ["monoview", "multiview"]` (:base_source:`l43 <summit/examples/config_files/config_example_1.yml#L43>`) allows for monoview and multiview algorithms to be used in the benchmark,
    - :yaml:`algos_monoview: ["decision_tree"]` (:base_source:`l45 <summit/examples/config_files/config_example_1.yml#L45>`) runs a Decision tree on each view,
    - :yaml:`algos_multiview: ["weighted_linear_late_fusion"]` (:base_source:`l47 <summit/examples/config_files/config_example_1.yml#L47>`) runs a late fusion,
    - The metrics configuration (:base_source:`l52-55 <summit/examples/config_files/config_example_1.yml#L52>`) ::

                        metrics:
                          accuracy_score:{}
                          f1_score:
                            average:"micro"

      means that the benchmark will evaluate the performance of each algorithms on accuracy, and f1-score with a micro average (because of the multi-class setting).

**Start the benchmark**

During the whole benchmark, the log file will be printed in the terminal. To start the benchmark, run :

.. code-block:: python

   >>> execute('example 1')

The execution should take less than one minute. We will first analyze the results and parse through the information the platform outputs.


**Understanding the results**

The result structure can be startling at first, but, as the platform provides a lot of information, it has to be organized.

The results are stored in :base_source:`a directory <summit/examples/results/example_1/>`. Here, you will find a directory with the name of the database used for the benchmark, here : ``summit_doc/``

Finally, a directory with the date and time of the beginning of the experiment. Let's say you started the benchmark on the 25th of December 1560, at 03:42 PM, the directory's name should be ``started_1560_12_25-15_42/``.

From here the result directory has the structure that follows  :

.. raw:: html
    :file: ./images/example_1/folder_description.html



The structure may seem complex, but it provides a lot of information, from the most general to the most precise.

Let's comment each file :

``*-accuracy_score*.html``, ``*-accuracy_score*.png`` and ``*-accuracy_score*.csv``
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

These files contain the scores of each classifier for the accuracy metric, ordered with the worst ones on the left and the best ones on the right, as an interactive html page, an image or a csv matrix. The star after ``accuracy_score*`` means that it was the principal metric (the usefulness of the principal metric will be explained later).
The html version is as follows :

.. raw:: html
    :file: ./images/example_1/accuracy.html

This is a bar plot showing the score on the training set (light gray), and testing set (black) for each monoview classifier on each view and or each multiview classifier.

Here, the generated dataset is build to introduce some complementarity amongst the views. As a consequence, the multiview algorithm has better performance than the monoview ones, but the difference is not that clear, which means that the hyper-parameters are maybe not optimal.

The ``.csv`` file is a matrix with the score on train stored in the first row and the score on test stored in the second one. Each classifier is presented in a column. It is loadable with pandas.

A similar graph ``*-accuracy_score*-class.html``, reports the error of each classifier on each class.

.. raw:: html
    :file: ./images/example_1/accuracy_class.html

Here, for each classifier, 8 bars are plotted, one for each class. It is clear that for the monoview algorithms, in view 2, the third class is difficult, as showed in the error matrix. However, the results show some difference with the error matrix. This sould be due to a specific train-test split. We will see alter how to avoid this issue.


``*-error_analysis_2D.png`` and ``*-error_analysis_2D.html``
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

In these files, one can visualize the success or failure of each classifier on each sample.

Below, ``*-error_analysis_2D.html`` is displayed.

It is the representation of a matrix, where the rows are the samples, and the columns are the classifiers.

The samples labelled as ``Mutual_error_*`` are mis-classified by most of the algorithms, the redundant ones are well-classified, and the complementary ones are mixly classified.

.. note::
    It is highly recommended to zoom in the html figure to see each row.

.. raw:: html
    :file: ./images/example_1/error_2d.html



This figure is the html version of the classifiers errors' visualization. It is interactive, so, by hovering it, the information on
each classifier and sample is printed. The classifiers are ordered as follows:

From left to right : all the monoview classifiers on the first view, all the ones on the second one, ..., then at the far right, the multiview classifiers

This html image is also available in ``.png`` format, but is then not interactive, so harder to analyze.

In terms of information, this is useful to detect possible outlier samples in the dataset and failing classifers.

For example, a mainly black horizontal line for an sample means that it has been missclassified by most of the classifiers.
It could mean that the sample is incorrectly labeled in the dataset or is very hard to classify.

Symmetrically, a mainly-black column means that a classifier spectacularly failed.

The data used to generate this matrix is available in ``*-2D_plot_data.csv``

``*-error_analysis_bar.png`` and ``*-error_analysis_bar.html``
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

This file is a different way to visualize the same information as the two previous ones. Indeed, it is a bar plot, with a bar for each sample, counting the ratio of classifiers that failed to classify this particular sample.

.. raw:: html
    :file: ./images/example_1/bar.html

All the spikes are the mutual error samples, the complementary ones are the 0.33 bars and the redundant are the empty spaces.

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

.. include:: ./images/example_1/summary.txt
    :literal:

This regroups the useful information on the classifier's configuration and it's performance. An interpretation section is available for classifiers that present some interpretation-related information (as feature importance).