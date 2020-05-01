import unittest
import numpy as np

import summit.multiview_platform.multiview_classifiers.additions.jumbo_fusion_utils as ju


class FakeDataset():

    def __init__(self, views, labels):
        self.nb_views = views.shape[0]
        self.dataset_length = views.shape[2]
        self.views = views
        self.labels = labels

    def get_v(self, view_index, sample_indices):
        return self.views[view_index, sample_indices]

    def get_nb_class(self, sample_indices):
        return np.unique(self.labels[sample_indices])


# TODO
