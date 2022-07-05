==================================================
Example 0 : Getting started with |platf| on MNist
==================================================

In the example, we will run |platf| on a famous dataset : `MNist <http://yann.lecun.com/exdb/mnist/>`_, available, already formatted :base_source:`in the project <summit/examples/data/>` .

Even if MNist is a monoview dataset, we adapted it to be multiview. The :base_source:`multiview_mnist.hdf5 <summit/examples/data/>` file contains 500 randomly selected samples from each label of the original MNist training set on which we computed a 12-direction Histogram of Oriented Gradients. Then, to create the views, we separated the directions in four randomly chosen groups of three directions.

So each view is comprised of 3 directions of a 12 direction HOG that was computed on 5000 samples.


Running |platf| on MNist
-------------------------

To run |platf| on MNist, we have to run :

.. code-block:: python

   >>> from summit.execute import execute
   >>> execute("example 0")

This will start a benchmark with this :base_source:`config file <summit/examples/config_files/config_example_0.yml>`.

By running |platf| with this configuration, one runs :

- one decision tree on each view,
- one early fusion decision tree multiview classifier that learns on the concatenation of all the views,
- one late fusion decision tree classifier that learns one monoview classifier by view and aggregates their decisions in a naive majority vote.



First discovery of the main results
-----------------------------------

The results will be saved in :base_source:`this directory <summit/examples/results/example_0/multiview_mnist/>`.

In this basic tutorial, we will only investigate the two main result files, however, during the other ones all the information given by the platform will be explained.

Getting the scores
<<<<<<<<<<<<<<<<<<

The main result returned by |platf| is the scores of each classifier. For this example, we only asked for the `accuracy <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html>`_ and `f1-score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score>`_, but |platf| can work with many other metrics.

The file that outputs the accuracy scores of all the classifiers is available in three versions :

- :base_source:`a png image <summit/examples/results/example_0/multiview_mnist/result_example/multiview_mnist-accuracy_score_p.png>`,
- :base_source:`a csv table <summit/examples/results/example_0/multiview_mnist/result_example/multiview_mnist-accuracy_score_p.csv>`,
- and an :base_source:`an html interactive file <summit/examples/results/example_0/multiview_mnist/result_example/multiview_mnist-accuracy_score_p.html>` :

.. raw:: html
    :file: images/example_0/acc.html


These three files contain the same information : the two figures are bar plots of the score of each classifier with the score on the training set in light gray and the score on the testing set in black.

Similarly, the f1-scores are saved in :base_source:`png <summit/examples/results/example_0/multiview_mnist/result_example/multiview_mnist-f1_score.png>`, :base_source:`csv <summit/examples/results/example_0/multiview_mnist/result_example/multiview_mnist-f1_score.csv>` and :base_source:`html <summit/examples/results/example_0/multiview_mnist/result_example/multiview_mnist-f1_score.html>`

.. raw:: html
    :file: images/example_0/f1.html

With these results, we are able to assess which classifier performs the best, here both the fusions have interesting scores compared to their monoview counterparts.


Getting more information on the classification
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

Once one knows the scores of each classifier, an interesting analysis could be to verify on which samples each classifier failed, to detect potential outliers.

This is possible with another result analysis, available in :base_source:`png <summit/examples/results/example_0/digits/result_example/digits-error_analysis_2D.png>`, :base_source:`csv <summit/examples/results/example_0/digits/result_example/digits_2D_plot_data.csv>` and :base_source:`html <summit/examples/results/example_0/digits/result_example/digits-error_analysis_2D.html>` :

.. raw:: html
    :file: images/example_0/err.html

This figure represents a matrix, with the samples in rows and classifiers in columns, with a white rectangle on row i, column j if classifier j succeeded to classify sample i.

.. note::
    To zoom on the image use your mouse to either draw a rectangle or drag it in a unique direction to zoom on an axis :

    .. image:: images/example_0/zoom_plotly.gif
        :scale: 100
        :align: center


A quick analysis of it shows that a decision tree (DT) on the view ``digit_col_grad_0`` is unable to classify any sample of labels 1, 2, 3 or 4. That both the other DTs have a similar behavior with other labels.

Concerning the fusions, if you zoom in on the samples labelled "2"", you may see that some errors made by the early fusion classifier are on samples that were mis-classified by the three DTs :

.. image:: images/example_0/lab_2.png
    :scale: 100
    :align: center


Conclusion
----------

Thanks to |platf| we were able to get a benchmark of mono- and multiview algorithms on a classification task.
In the following tutorials, we will develop the features of |platf| on several samples.