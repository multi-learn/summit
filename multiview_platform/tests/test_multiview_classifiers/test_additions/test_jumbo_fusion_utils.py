import unittest
import numpy as np

import  multiview_platform.mono_multi_view_classifiers.multiview_classifiers.additions.jumbo_fusion_utils  as ju


class FakeDataset():

    def __init__(self, views, labels):
        self.nb_views = views.shape[0]
        self.dataset_length = views.shape[2]
        self.views = views
        self.labels = labels

    def get_v(self, view_index, example_indices):
        return self.views[view_index, example_indices]

    def get_nb_class(self, example_indices):
        return np.unique(self.labels[example_indices])


#TODO