=============================
Example 0 : |platf| on digits
=============================

In the example, we will run |platf| on a famous dataset : `digits <https://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html>`_, :base_source:`in the project <multiview_platform/examples/data/>` .

TODO.

Running |platf|
---------------

To run |platf| on digits, one has to run

.. code-block:: python

   >>> from multiview_platform.execute import execute
   >>> execute("example 0")

This will start a benchmark with this :base_source:`config file <multiview_platform/examples/config_files/config_example_0.yml>`.

By runnint |platf| with this configuration, one runs :

- One decision tree on each view,
- One early fusion multiview classifier that learns on the concatenation of all the views,
- One late fusion classifier that learn one monoview lassifier by view and aggregates their decisions in a majority vote.



The results
-----------

The results will be saved in :base_source:`this directory <multiview_platform/examples/results/example_0/>`.

In this basic tutorial, we will only investigate the two main result files, however, during the other ones all the information given by the platform will be explained.

Getting the scores
<<<<<<<<<<<<<<<<<<

The main result returned by |platf| is the scores of each classifier. For this example, we only asked for the `accuracy <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html>`_ and `f1-score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score>`_, but |platf| can work with many other metrics.

Accuracy
>>>>>>>>

The file that regroups the accuracy scores is available in three versions :

- a png image,
- a csv table,
- and an html interactive file :

.. raw:: html
    :file: images/example_0/acc.html
