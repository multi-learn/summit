.. |pipeline| image:: https://gitlab.lis-lab.fr/baptiste.bauvin/summit/badges/master/pipeline.svg
    :alt: Pipeline status

.. |license| image:: https://img.shields.io/badge/License-New%20BSD-blue.svg
    :target: https://opensource.org/licenses/BSD-3-Clause
    :alt: License: New BSD

.. |coverage| image:: https://gitlab.lis-lab.fr/baptiste.bauvin/summit/badges/master/coverage.svg
    :target: http://baptiste.bauvin.pages.lis-lab.fr/summit/coverage/index.html
    :alt: Coverage

|pipeline| |license| |coverage|




Supervised MultiModal Integration Tool's Readme
===============================================

This project aims to be an easy-to-use solution to run a prior benchmark on a dataset and evaluate mono- & multi-view algorithms capacity to classify it correctly.

Getting Started
---------------

SuMMIT has been designed and uses continuous integration for Linux platforms (ubuntu 18.04), but we try to keep it as compatible as possible with Mac and Windows.

+----------+-------------------+
| Platform | Last positive test|
+==========+===================+
|   Linux  |  |pipeline|       |
+----------+-------------------+
| Mac      | 1st of May, 2020  |
+----------+-------------------+
| Windows  | 1st of May, 2020  |
+----------+-------------------+


Prerequisites
<<<<<<<<<<<<<

To be able to use this project, you'll need :

* `Python 3 <https://docs.python.org/3/>`_

And the following python modules will be automatically installed  :

* `numpy <http://www.numpy.org/>`_, `scipy <https://scipy.org/>`_,
* `matplotlib <http://matplotlib.org/>`_ - Used to plot results,
* `sklearn <http://scikit-learn.org/stable/>`_ - Used for the monoview classifiers,
* `joblib <https://pypi.python.org/pypi/joblib>`_ - Used to compute on multiple threads,
* `h5py <https://www.h5py.org>`_ - Used to generate HDF5 datasets on hard drive and use them to spare RAM,
* `pickle <https://docs.python.org/3/library/pickle.html>`_ - Used to store some results,
* `pandas <https://pandas.pydata.org/>`_ - Used to manipulate data efficiently,
* `six <https://pypi.org/project/six/>`_ -
* `m2r <https://pypi.org/project/m2r/>`_ - Used to generate documentation from the readme,
* `docutils <https://pypi.org/project/docutils/>`_ - Used to generate documentation,
* `pyyaml <https://pypi.org/project/PyYAML/>`_ - Used to read the config files,
* `plotly <https://plot.ly/>`_ - Used to generate interactive HTML visuals,
* `tabulate <https://pypi.org/project/tabulate/>`_ - Used to generated the confusion matrix.
* `pyscm-ml <https://pypi.org/project/pyscm-ml/>`_ - 


Installing
<<<<<<<<<<

Once you cloned the project from the `github repository <https://github.com/multi-learn/summit/>`_, you just have to use :

.. code:: bash

    cd path/to/summit/
    pip install -e .


In the `summit` directory to install SuMMIT and its dependencies.

Running the tests
<<<<<<<<<<<<<<<<<

To run the test suite of SuMMIT, run :

.. code:: bash

    cd path/to/summit
    pip install -e .[dev]
    pytest

The coverage report is automatically generated and stored in the ``htmlcov/`` directory

Building the documentation
<<<<<<<<<<<<<<<<<<<<<<<<<<

To locally build the `github-documentation <https://multi-learn.github.io/summit/index.html>`_ run :

.. code:: bash

    cd path/to/summit
    pip install -e .[doc]
    python setup.py build_sphinx

The built html files will be stored in ``path/to/summit/build/sphinx/html``

Running on simulated data
<<<<<<<<<<<<<<<<<<<<<<<<<

For your first go with SuMMIT, you can run it on simulated data with

.. code:: bash

    python
    >>> from summit.execute import execute
    >>> execute("example 1")

This will run the benchmark of `documentation's Example 1 <http://baptiste.bauvin.pages.lis-lab.fr/summit/tutorials/example1.html>`_.

For more information about the examples, see the `documentation <http://baptiste.bauvin.pages.lis-lab.fr/summit/index.html>`_.
Results will, by default, be stored in the results directory of the installation path :
``path/to/summit/multiview_platform/examples/results``.

The documentation proposes a detailed interpretation of the results and arguments of SuMMIT through `6 tutorials <http://baptiste.bauvin.pages.lis-lab.fr/summit/>`_.

Dataset compatibility
<<<<<<<<<<<<<<<<<<<<<

In order to start a benchmark on your own dataset, you need to format it so SuMMIT can use it. To do so, a `python script <https://gitlab.lis-lab.fr/baptiste.bauvin/summit/-/blob/master/format_dataset.py>`_ is provided.

For more information, see `Example 5 <http://baptiste.bauvin.pages.lis-lab.fr/summit/tutorials/example5.html>`_

Running on your dataset
+++++++++++++++++++++++

Once you have formatted your dataset, to run SuMMIT on it you need to modify the config file as

.. code:: yaml

    name: ["your_file_name"]
    pathf: "path/to/your/dataset"


It is however highly recommended to follow the documentation's `tutorials <http://baptiste.bauvin.pages.lis-lab.fr/summit/tutorials/index.html>`_ to learn the use of each parameter.
 

Authors
-------

* **Baptiste BAUVIN**
* **Dominique BENIELLI**
* **Alexis PROD'HOMME**

