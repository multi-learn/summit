.. |algo| replace:: name_me

========================================
Taking control : Use your own algorithms
========================================

One of the main goals of this platform is to be able to add a classifier to it without modifying the main code.

Simple task : Adding a monoview classifier
------------------------------------------

Make it work
<<<<<<<<<<<<

Let's say we want to add a monoview classifier called "|algo|" to the platform in order to compare it to the other available ones.
Let's suppose that we have a python module ``algo_module.py`` in which |algo| is defined in the class :python:`Algo` following ``scikit-learn``'s `guidelines <https://scikit-learn.org/stable/developers/index.html>`_ .

To add "algo" to the platform, let's create a file called ``algo.py`` in ``multiview_platform/mono_multi_view_classifiers/monoview_classifiers/``

In this file let's define the class :python:`AlgoClassifier`, inheriting from :python:`Algo` and :base_source:`BaseMonoviewClassifier <multiview_platform/mono_multi_view_classifiers/monoview/monoview_utils.py#L115>` that contains the required methods for |platf|.

Moreover, one has to add a variable called :python:`classifier_class_name` that contains the class name (here ``'AlgoClassifier'``)

.. code-block:: python

    import Algo
    from ..monoview.monoview_utils import BaseMonoviewClassifier

    classifier_class_name = "AlgoClassifier"

    class AlgoClassifier(Algo, BaseMonoviewClassifier):


To be able to use the randomized hyper-parameter optimization, we need to provide some information in the :python:`__init__()` method.
Indeed, all the algorithms included in the platform must provide two hyper-parameter-related attributes :

- :python:`self.param_names` that contain the name of the hyper-parameters that have to be optimized (they must correspond to the name of the attributes of the class :python:`Algo`)
- :python:`self.distribs` that contain the distributions for each of these hyper-parameters.

For example, let's suppose that |algo| need three hyper-parameters and a random state parameter allowing reproducibility :

- :python:`trade_off` that is a float between 0 and 1,
- :python:`norm_type` that is a string in :python:`["l1", "l2"]`,
- :python:`max_depth` that is an integer between 0 and 100.

Then, the :python:`__init__()` method of the :python:`AlgoClassifier` class wil be :

.. code-block:: python

    import Algo
    from ..monoview.monoview_utils import BaseMonoviewClassifier, CustomUniform, CustomRandint

    classifier_class_name = "AlgoClassifier"

    class AlgoClassifier(Algo, BaseMonoviewClassifier):

        def __init__(self, random_sate=42, trade_off=0.5, norm_type='l1', max_depth=50)

            super(AlgoClassifier, self).__init__(random_sate=random_sate,
                                                 trade_off=trade_off,
                                                 norm_type=norm_type,
                                                 max_depth=max_depth)

            self.param_names = ["trade_off", "norm_type", "max_depth"]
            self.distribs = [CustomUniform(),
                             ["l1", "l2"],
                             CustomRandint()]

In this method, we added the needed attributes. See REF TO DOC OF DISTRIBS for the documentation on the used distributions.

If "algo" is implemented in a sklearn fashion, it is now usable in the platform.

Interpretation
<<<<<<<<<<<<<<

It is possible to provide some information about the decision process of the algorithm in the :python:`get_interpretation` method.

It inputs four arguments :

* :python:`directory`, a string containing the directory where figures should be sotred
* :python:`base_file_name`, a string containing the file name prefix that should be used to store figures
* :python:`y_test`, an array containing the labels of the test set
* :python:`multiclass` a boolean that is True if the target is multiclass

This method must return a string that will be appended to the summary file.

An example of method can be :

.. code-block:: python

    def get_interpretation(self, directory, base_file_name, y_test,
                           multiclass=False):
        interpret_string = "Algo is a very relevant algorithm that used all the features to classify"
        # Save a figure in os.path.join(directory, base_file_name+figure_name.png")
        return interpretString


More complex task : Adding a multiview classifier
-------------------------------------------------

This part is a bit more complex as to the best of our knowledge, there is no consensus regarding a multiview input for a classifier.

The first step of the integration of a multiview classifier is very similar to the monoview one let us suppose one wants to add "new mv algo", that is implemented in the class `NewMVAlgo`. To do so, create a "new_mv_algo.py" file in ``multiview_platform/mono_multi_view_classifiers/multiview_classifiers/``.

In this file let's define the class :python:`NewMVAlgoClassifier`, inheriting from :python:`NewMVAlgo` and :base_source:`BaseMultiviewClassifier <multiview_platform/mono_multi_view_classifiers/multiview/multiview_utils.py#L16>` that contains the required methods for the platform.

Moreover, one has to add a variable called :python:`classifier_class_name` that contains the class name (here ``'NewMVAlgoClassifier'``)

.. code-block:: python

    from new_mv_algo_module import NewMVAlgo
    from ..multiview.multiview_utils import BaseMultiviewClassifier

    from ..utils.hyper_parameter_search import CustomRandint

    classifier_class_name = "NewMVAlgoClassifier"

    class NewMVAlgoClassifier(BaseMultiviewClassifier, NewMVAlgo):

        def __init__(self, param_1=50,
                         random_state=None,
                         param_2="edge"):
                BaseMultiviewClassifier.__init__(self, random_state)
                NewMVAlgo.__init__(self, param_1=param_1,
                                            random_state=random_state,
                                            param_2=param_2)
                self.param_names = ["param_1", "random_state", "param_2"]
                self.distribs = [CustomRandint(5,200), [random_state], ["val_1", "val_2"]]

In |platf| the input of the :python:`fit()` method is `X`, a dataset object that provide access to each view with a method : :python:`dataset_var.get_v(view_index, example_indices)`.
So in order to add a mutliview classifier to |platf|, one will probably have to add a data-transformation step before using the class's :python:`fit()` method.

Moreover, to get restrain the examples and descriptors used in the method, |platf| provides two supplementary arguments :

- ``train_indices`` is an array of examples indices that compose the training set,
- ``view_indices`` is an array of view indices to restrain the number of views on which the algorithm will train.

These two arguments are useful to reduce memory usage. Indeed, `X`, the dataset object is just a wrapper for an HDF5 file object, so the data will only be loaded once the `get_v` method is called, so the train and test set are not loaded at the same time.



.. code-block:: python

    def fit(self, X, y, train_indices=None, view_indices=None):
        # This function is used to initialize the example and view indices, in case they are None, it transforms them in the correct values
        train_indices, view_indices = get_examples_views_indices(X,
                                                                 train_indices,
                                                                 view_indices)
        needed_input = transform_data_if_needed(X, train_indices, view_indices)
        return NewMVAlgo.fit(self, needed_input, y[train_indices])

    def predict(self, X, example_indices=None, view_indices=None):
        example_indices, view_indices = get_examples_views_indices(X,
                                                                 example_indices,
                                                                 view_indices)
        needed_input = transform_data_if_needed(X, example_indices, view_indices)
        return NewMVAlgo.predict(self, needed_input)

Similarly to monoview algorithms, it is possible to add an interpretation method.

Manipulate the dataset object
-----------------------------

The input of the fit and predict method is a :base_source:`Dataset object  <multiview_platform/mono_multi_view_classifiers/utils/dataset.py#L13>`.

The useful methods of this object are

:base_source:`get_v <multiview_platform/mono_multi_view_classifiers/utils/dataset.py#L360>`
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

The :base_source:`get_v <multiview_platform/mono_multi_view_classifiers/utils/dataset.py#L360>` method is **the** way to access the view data in the dataset object.

As explained earlier, |platf| communicates the **full** dataset object and two arrays through the :python:`fit()` and :python:`predict()` methods to avoid loading the views if it is not mandatory.

Example : build a list of all the views arrays
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Let us suppose that the mutliview algorithm that one wants to add to |platf| takes as input a list :python:`list_X` of all the views.

Then an example of :python:`self.transform_data_if_needed(X, example_indices, view_indices)` could be :

.. code-block:: python

    def transform_data_if_needed(self, X, example_indices, view_indices):
        views_list = []
        # Browse the asked views indices
        for view_index in view_indices:
            # Get the data from the dataset object, for the asked examples
            view_data = X.get_v(view_index, example_indices=example_indices)
            # Store it in the list
            views_list.append(view_data)
        return views_list


            


