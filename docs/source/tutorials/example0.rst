==================================================
Example 0 : Getting started with |platf| on digits
==================================================

In the example, we will run |platf| on a famous dataset : `digits <https://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html>`_, available, already formatted :base_source:`in the project <multiview_platform/examples/data/>` .

TODO.

Running |platf| on digits
-------------------------

To run |platf| on digits, one has to run :

.. code-block:: python

   >>> from multiview_platform.execute import execute
   >>> execute("example 0")

This will start a benchmark with this :base_source:`config file <multiview_platform/examples/config_files/config_example_0.yml>`.

By running |platf| with this configuration, one runs :

- One decision tree on each view,
- One early fusion multiview classifier that learns on the concatenation of all the views,
- One late fusion classifier that learn one monoview lassifier by view and aggregates their decisions in a majority vote.



First discovery of the main results
-----------------------------------

The results will be saved in :base_source:`this directory <multiview_platform/examples/results/example_0/digits/>`.

In this basic tutorial, we will only investigate the two main result files, however, during the other ones all the information given by the platform will be explained.

Getting the scores
<<<<<<<<<<<<<<<<<<

The main result returned by |platf| is the scores of each classifier. For this example, we only asked for the `accuracy <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html>`_ and `f1-score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score>`_, but |platf| can work with many other metrics.

The file that regroups the accuracy scores is available in three versions :

- :base_source:`a png image <multiview_platform/examples/results/example_0/digits/result_example/digits-accuracy_score*.png>`,
- :base_source:`a csv table <multiview_platform/examples/results/example_0/digits/result_example/digits-accuracy_score*.csv>`,
- and an :base_source:`an html interactive file <multiview_platform/examples/results/example_0/digits/result_example/digits-accuracy_score*.html>` :

.. raw:: html
    .. :file: images/example_0/acc.html
    :file: images/fake.html

These three files contain the same information : the two figures are bar plots of the score of each classifier with the score on the training set in light gray and the score on the testing set in black.

Similarly, the f1-scores are saved in :base_source:`png <multiview_platform/examples/results/example_0/digits/result_example/digits-f1_score.png>`, :base_source:`csv <multiview_platform/examples/results/example_0/digits/result_example/digits-f1_score.csv>` and :base_source:`html <multiview_platform/examples/results/example_0/digits/result_example/digits-f1_score.html>`

With these results, we are able to assess which classifier perfroms the best, here both the fusions have interesting scores compared to their monoview counterparts.


Getting more information on the classification
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

Once one has the scores of each classifier, an interesting analysis could be to verify on which examples each classifier failed, to detect potential outliers.

This is possible with another result analysis, available in :base_source:`png <multiview_platform/examples/results/example_0/digits/result_example/digits-error_analysis_2D.png>`, :base_source:`csv <multiview_platform/examples/results/example_0/digits/result_example/digits_2D_plot_data.csv>` and :base_source:`html <multiview_platform/examples/results/example_0/digits/result_example/digits-error_analysis_2D.html>` :

.. raw:: html
    .. :file: images/example_0/err.html
    :file: images/fake.html

This figure represents a matrix, with the examples in rows and classifiers in columns, with a white rectangle on row i, column j if classifier j failed to classify example i.

A quick analysis of it shows that a decision tree (DT) on the view ``digit_col_grad_0`` is unable to classify any example of labels 1, 2, 3 or 4. That both the other DTs have a similar behavior with other labels.
Concerning the fusions, if you zoom in on the examples labelled "2"", you may see that some errors made by the early fusion classifier are on examples that were mis-classified by the three DTs :

.. image:: images/example_0/lab_2.png
    :scale: 100
    :align: center


Conclusion
----------

TODO