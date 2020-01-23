.. |algo| replace:: name_me
========================================
Taking control : Use your own algorithms
========================================

.. role:: python(code)
    :language: python

One of the main goals of this platform is to be able to add a classifier to it without modifying the code.

Simple task : Adding a monoview classifier
------------------------------------------

Let's say we want to add a monoview classifier called "algo" to the platform in order to compare it to the other available ones.
Let's suppose that we have a python module ``algo_module.py`` in which algo is defined in the class :python:`Algo` with the guidelines of ``sklearn``.

To add algo to the platform, let's create a file called ``algo.py`` in ``multiview_platform/mono_multi_view_classifiers/monoview_classifiers/``

In this file let's define the class :python:`AlgoClassifier`, inheriting from :python:`Algo` and :python:`BaseMonoviewClassifier` that contains the required methods for the platfrom.

.. code-block:: python

    import Algo
    from ..monoview.monoview_utils import BaseMonoviewClassifier

    class AlgoClassifier(Algo, BaseMonoviewClassifier):


To be able to use the hyper-parameter optimization of the platform, we need to provide some information in the :python:`__init__()` method.
Indeed, all the algorithms included in the platform must provide two hyper-parameter-related attributes :

- :python:`self.param_names` that contain the name of the hyper-parameters that have to be optimized (they must correspond to the name of the attributes of the class :python:`Algo`)
- :python:`self.distribs` that contain the distributions for each of these hyper-parameters.

For example, let's suppose that algo need three hyper-parameters and a random state parameter that allow reproducibility :

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

In this method, we added the needed attributes. See REF TO DOC OF DISTRIBS for the dicumentation on the used distributions.

If "algo" is implemented in a sklearn fashion, it is now usable in the platform.

TODO interpretation

More complex task : Adding a multiview classifier
-------------------------------------------------



.. code-block:: python

    from mimbo import MimboClassifier
    from ..multiview.multiview_utils import BaseMultiviewClassifier, \
                                            get_examples_views_indices
    from ..utils.hyper_parameter_search import CustomRandint

    classifier_class_name = "Mimbo"

    class Mimbo(BaseMultiviewClassifier, MimboClassifier):

        def __init__(self, n_estimators=50,
                     random_state=None,
                     best_view_mode="edge"):
            super().__init__(random_state)
            super(BaseMultiviewClassifier, self).__init__(n_estimators=n_estimators,
                                        random_state=random_state,
                                        best_view_mode=best_view_mode)
            self.param_names = ["n_estimators", "random_state", "best_view_mode"]
            self.distribs = [CustomRandint(5,200), [random_state], ["edge", "error"]]

        def fit(self, X, y, train_indices=None, view_indices=None):
            train_indices, view_indices = get_examples_views_indices(X,
                                                                     train_indices,
                                                                     view_indices)
            numpy_X, view_limits = X.to_numpy_array(example_indices=train_indices,
                                                    view_indices=view_indices)
            return super(Mimbo, self).fit(numpy_X, y[train_indices],
                                                    view_limits)

        def predict(self, X, example_indices=None, view_indices=None):
            example_indices, view_indices = get_examples_views_indices(X,
                                                                     example_indices,
                                                                     view_indices)
            numpy_X, view_limits = X.to_numpy_array(example_indices=example_indices,
                                                    view_indices=view_indices)
            return super(Mimbo, self).predict(numpy_X)

