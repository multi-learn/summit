# -*- coding: utf-8 -*-

"""This module contains the DataSample class and Splearn_array class
The DataSample class encapsulates a sample 's components
nbL and nbEx numbers,
Splearn_array class inherit from numpy ndarray and contains a 2d data ndarray
with the shape

==== ====  ====  ====  ====
x    x     x     x     -1
x    x     x     x     x
x    x     -1    -1    -1
x    -1    -1    -1    -1
-1   -1    -1    -1    -1
==== ====  ====  ====  ====

where -1 a indicates a empty cell,
the number nbL and nbEx and , the fourth dictionaries for sample,
prefix, suffix and factor where they are computed
"""
import numpy as np
import numpy.ma as ma


class MultiView_array(ma.MaskedArray):
    """Splearn_array inherit from numpy ndarray

    :Example:

    >>> from multiview_platform.datasets.base import load_data
    >>> from multiview_platform.datasets.get_dataset_path import get_dataset_path
    >>> train_file = '' # '4.spice.train'
    >>> data = load_data(adr=get_dataset_path(train_file))
    >>> print(data.__class__)
    >>> data.data

    """
    def __new__(cls, data):
        shapes_int = []
        index = 0
        new_data = data
        shape_ext = len(data)
        thekeys = None
        if isinstance(data, dict):
            shape_ext = len(data)
            for key, dat_values in data.items():
                new_data = cls._populate_new_data(index, dat_values, new_data)
                shapes_int.append(dat_values.shape[0])
                index += 1
            thekeys = data.keys()

        if isinstance(data, np.ndarray):
            shape_ext = data.shape[0]
            for dat_values in data:
                shapes_int.append(dat_values.shape[0])
                new_data = cls._populate_new_data(index, dat_values, new_data)
                index += 1
        # obj =   ma.MaskedArray.__new(new_data)   # new_data.view()  a.MaskedArray(new_data, mask=new_data.mask).view(cls)
        # bj = super(Metriclearn_array, cls).__new__(cls, new_data.data, new_data.mask)
        obj = ma.masked_array(new_data.data, new_data.mask).view(cls)
        obj.shapes_int = shapes_int
        obj.shape_ext = shape_ext
        obj.keys = thekeys
        return obj


    @staticmethod
    def _populate_new_data(index, dat_values, new_data):
        if index == 0:
            if isinstance(dat_values, ma.MaskedArray):
                new_data = dat_values
            else:
                new_data = dat_values.view(ma.MaskedArray) #  ma.masked_array(dat_values, mask=ma.nomask) dat_values.view(ma.MaskedArray) #(
                new_data.mask = ma.nomask
        else:
            if isinstance(dat_values, ma.MaskedArray):
                new_data = ma.hstack((new_data, dat_values))
            else:
                new_data = ma.hstack((new_data,  dat_values.view(ma.MaskedArray) ) ) #  ma.masked_array(dat_values, mask=ma.nomask
        return new_data

    def __array_finalize__(self, obj):
        if obj is None: return
        super(MultiView_array, self).__array_finalize__(obj)
        self.shapes_int = getattr(obj, 'shapes_int', None)
        self.shape_ext = getattr(obj, 'shape_ext', None)
        self.keys = getattr(obj, 'keys', None)

    def getCol(self, view, col):
        start = np.sum(np.asarray(self.shapes_int[0: view]))
        return self.data[start+col, :]

    def getView(self, view):
        start = np.sum(np.asarray(self.shapes_int[0: view]))
        stop = start + self.shapes_int[view]
        return self.data[start:stop, :]

    def getRaw(self, view, raw):
        start = np.sum(np.asarray(self.shapes_int[0: view]))
        stop = np.sum(np.asarray(self.shapes_int[0: view+1]))
        return self.data[start:stop, raw]

class DataSample(dict):
    """ A DataSample instance

    :Example:

    >>> from multiview_platform.datasets.base import load_data
    >>> from multiview_platform.datasets.get_dataset_path import get_dataset_path
    >>> train_file = '' # '4.spice.train'
    >>> data = load_data_sample(adr=get_dataset_path(train_file))
    >>> print
    (data.__class__)

    >>> data.data

    - Input:

    :param string adr: adresse and name of the loaden file
    :param string type: (default value = 'SPiCe') indicate
           the structure of the file
    :param lrows: number or list of rows,
           a list of strings if partial=True;
           otherwise, based on self.pref if version="classic" or
           "prefix", self.fact otherwise
    :type lrows: int or list of int
    :param lcolumns: number or list of columns
           a list of strings if partial=True ;
           otherwise, based on self.suff if version="classic" or "suffix",
           self.fact otherwise
    :type lcolumns: int or list of int
    :param string version: (default = "classic") version name
    :param boolean partial: (default value = False) build of partial

    """

    def __init__(self, data=None, **kwargs):

        # The dictionary that contains the sample
        super(DataSample, self).__init__(kwargs)
        self._data = None # Metriclearn_array(np.zeros((0,0)))
        if data is not None:
            self._data = MultiView_array(data)


    @property
    def data(self):
        """Metriclearn_array"""

        return self._data

    @data.setter
    def data(self, data):
        if isinstance(data, (MultiView_array, np.ndarray, ma.MaskedArray, np.generic)):
            self._data = data
        else:
            raise TypeError("sample should be a MultiView_array.")




