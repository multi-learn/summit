import numpy as np

from multiview_platform.mono_multi_view_classifiers import monoview_classifiers
from .additions.fusion_utils import BaseFusionClassifier
from ..multiview.multiview_utils import get_available_monoview_classifiers, \
    BaseMultiviewClassifier, ConfigGenerator
from ..utils.dataset import get_examples_views_indices
from ..utils.multiclass import get_mc_estim, MultiClassWrapper

# from ..utils.dataset import get_v

classifier_class_name = "WeightedLinearEarlyFusion"


class WeightedLinearEarlyFusion(BaseMultiviewClassifier, BaseFusionClassifier):
    """
    WeightedLinearEarlyFusion

    Parameters
    ----------
    random_state
    view_weights
    monoview_classifier_name
    monoview_classifier_config

    Attributes
    ----------
    """

    def __init__(self, random_state=None, view_weights=None,
                 monoview_classifier_name="decision_tree",
                 monoview_classifier_config={}):
        BaseMultiviewClassifier.__init__(self, random_state=random_state)
        self.view_weights = view_weights
        self.monoview_classifier_name = monoview_classifier_name
        self.short_name = "early fusion " + self.monoview_classifier_name
        if monoview_classifier_name in monoview_classifier_config:
            self.monoview_classifier_config = monoview_classifier_config[
                monoview_classifier_name]
        self.monoview_classifier_config = monoview_classifier_config
        monoview_classifier_module = getattr(monoview_classifiers,
                                             self.monoview_classifier_name)
        monoview_classifier_class = getattr(monoview_classifier_module,
                                            monoview_classifier_module.classifier_class_name)
        self.monoview_classifier = monoview_classifier_class(
            random_state=random_state,
            **self.monoview_classifier_config)
        self.param_names = ["monoview_classifier_name",
                            "monoview_classifier_config"]
        self.distribs = [get_available_monoview_classifiers(),
                         ConfigGenerator(get_available_monoview_classifiers())]
        self.classed_params = []
        self.weird_strings = {}

    def set_params(self, monoview_classifier_name=None,
                   monoview_classifier_config=None, **params):
        self.monoview_classifier_name = monoview_classifier_name
        self.monoview_classifier = self.init_monoview_estimator(
            monoview_classifier_name,
            monoview_classifier_config)
        self.monoview_classifier_config = self.monoview_classifier.get_params()
        self.short_name = "early fusion " + self.monoview_classifier_name
        return self

    def get_params(self, deep=True):
        return {"random_state": self.random_state,
                "view_weights": self.view_weights,
                "monoview_classifier_name": self.monoview_classifier_name,
                "monoview_classifier_config": self.monoview_classifier_config}

    def fit(self, X, y, train_indices=None, view_indices=None):
        train_indices, X = self.transform_data_to_monoview(X, train_indices,
                                                           view_indices)
        if np.unique(y[train_indices]).shape[0] > 2 and \
                not (isinstance(self.monoview_classifier, MultiClassWrapper)):
            self.monoview_classifier = get_mc_estim(self.monoview_classifier,
                                                    self.random_state,
                                                    multiview=False,
                                                    y=y[train_indices])
        self.monoview_classifier.fit(X, y[train_indices])
        return self

    def predict(self, X, example_indices=None, view_indices=None):
        _, X = self.transform_data_to_monoview(X, example_indices, view_indices)
        predicted_labels = self.monoview_classifier.predict(X)
        return predicted_labels

    def transform_data_to_monoview(self, dataset, example_indices,
                                   view_indices):
        """Here, we extract the data from the HDF5 dataset file and store all
        the concatenated views in one variable"""
        example_indices, self.view_indices = get_examples_views_indices(dataset,
                                                                        example_indices,
                                                                        view_indices)
        if self.view_weights is None:
            self.view_weights = np.ones(len(self.view_indices), dtype=float)
        else:
            self.view_weights = np.array(self.view_weights)
        self.view_weights /= float(np.sum(self.view_weights))

        X = self.hdf5_to_monoview(dataset, example_indices)
        return example_indices, X

    def hdf5_to_monoview(self, dataset, examples):
        """Here, we concatenate the views for the asked examples """
        monoview_data = np.concatenate(
            [dataset.get_v(view_idx, examples)
             for view_weight, (index, view_idx)
             in zip(self.view_weights, enumerate(self.view_indices))]
            , axis=1)
        return monoview_data

    # def set_monoview_classifier_config(self, monoview_classifier_name, monoview_classifier_config):
    #     if monoview_classifier_name in monoview_classifier_config:
    #         self.monoview_classifier.set_params(**monoview_classifier_config[monoview_classifier_name])
    #     else:
    #         self.monoview_classifier.set_params(**monoview_classifier_config)
