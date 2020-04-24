from __future__ import print_function
import pickle
import numpy as np
import numpy.ma as ma
from multiview_platform.datasets.data_sample import DataSample
from six.moves import cPickle as pickle #for performance
import numpy as np


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


def load_data(address, output='array', pickle=False):
    if output.startswith(('array')):
        views = np.empty((len(address)), dtype=object)
    else:
        views = {}
    i = 0
    nb_samples, nb_features = _determine_dimensions(address)
    for addr in address:
        data = _load_view_sample(addr, nb_samples , nb_features[i], pickle=pickle)
        views[i] = data
        i += 1
    return DataSample(data=views)

def _determine_dimensions(address):
    nb_features = []
    nb_samples = 0
    nb_sample_max = -1
    for adr in address:
      try:
          f = open(adr, "r")
          line = f.readline()
          nb_samples += 1
          while line :
              line = f.readline()
              l = line.split()
              nb_samples += 1
              nb_features.append(len(l))
              line = f.readline()
          if nb_sample_max < nb_samples:
              nb_sample_max = nb_samples
          f.close()
      except IOError:
          raise IOError("file adr can't be open")
    return nb_sample_max, nb_features

def _load_view_sample(adr, nb_samples, nb_features, pickle=False):
    """Load a sample from file and returns a dictionary
    (word,count)

    - Input:

    :param lrows: number or list of rows,
           a list of strings if partial=True;
           otherwise, based on pref if version="classic" or
           "prefix", fact otherwise
    :type lrows: int or list of int
    :param lcolumns: number or list of columns
            a list of strings if partial=True ;
            otherwise, based on suff if version="classic" or "suffix",
            fact otherwise
    :type lcolumns: int or list of int
    :param string version: (default = "classic") version name
    :param boolean partial: (default value = False) build of partial
           if True partial dictionaries are loaded based
           on nrows and lcolumns

    - Output:

    :returns:  nbL , nbEx , dsample , dpref , dsuff  , dfact
    :rtype: int , int , dict , dict , dict  , dict


    :Example:

    Let's say you are interested in the samples 10, 25, and 50, and want to
    know their class name.

    >>> from metriclearning.datasets.base import load_data_sample
    >>> from metriclearning.tests.datasets.get_dataset_path import get_dataset_path
    >>> train_file = '3.pautomac_light.train' # '4.spice.train'
    >>> data = load_data_sample(adr=get_dataset_path(train_file))
    >>> data.nbL
    4
    >>> data.nbEx
    5000
    >>> data.data
    Splearn_array([[ 3.,  0.,  3., ..., -1., -1., -1.],
           [ 3.,  3., -1., ..., -1., -1., -1.],
           [ 3.,  2.,  0., ..., -1., -1., -1.],
           ...,
           [ 3.,  1.,  3., ..., -1., -1., -1.],
           [ 3.,  0.,  3., ..., -1., -1., -1.],
           [ 3.,  3.,  1., ..., -1., -1., -1.]])

    """
    #nb_sample, max_length = _read_dimension(adr=adr)
    f = open(adr, "r")
    line = f.readline()
    l = line.split()
    nbEx = int(l[0])
    nbL = int(l[1])
    line = f.readline()
    data1 = np.zeros((nb_samples, nb_features), dtype=np.float)
    data1 += np.NAN
    datatrue = np.ones((nb_samples, nb_features), dtype=np.bool)
    i = 0
    while line:
        l = line.split()
        # w = () if int(l[0]) == 0 else tuple([int(x) for x in l[1:]])
        # dsample[w] = dsample[w] + 1 if w in dsample else 1
        # traitement du mot vide pour les préfixes, suffixes et facteurs
        w = [float(x) for x in l[0:]]
        data1[i, :len(w)] = w
        line = f.readline()
        i += 1
        if i > nbEx:
            raise IndexError("dimension is not well defined")
    masint= np.isnan(data1)
    # masint = np.logical_not(masint)
    madata1 = ma.MaskedArray(data1, masint)
    f.close()

    if pickle:
        _create_pickle_files(adr=adr, dsample=madata1)
    return madata1

# def _read_dimension(adr):
#     f = open(adr, "r")
#     line = f.readline()
#     l = line.split()
#     nbEx = int(l[0])
#     nbL = int(l[1])
#     line = f.readline()
#     max_length = 0
#     nb_sample = 0
#     while line:
#         l = line.split()
#         nb_sample += 1
#         length = int(l[0])
#         if max_length < length:
#             max_length = length
#         line = f.readline()
#     f.close()
#     if nb_sample != nbEx:
#         raise ValueError("check imput file, metadata " + str(nbEx) +
#                          "do not match number of samples " + str(nb_sample))
#     return nb_sample , max_length

# def _load_file_1lecture(adr, pickle=False):
#     dsample = {}  # dictionary (word,count)
#     f = open(adr, "r")
#     line = f.readline()
#     l = line.split()
#     nbEx = int(l[0])
#     nbL = int(l[1])
#     line = f.readline()
#     data1 = np.zeros((0,0))
#     length = 0
#     while line:
#         l = line.split()
#         # w = () if int(l[0]) == 0 else tuple([int(x) for x in l[1:]])
#         # dsample[w] = dsample[w] + 1 if w in dsample else 1
#         # traitement du mot vide pour les préfixes, suffixes et facteurs
#         w = [] if int(l[0]) == 0 else [int(x) for x in l[1:]]
#         word = np.array(w, ndmin=2, dtype=np.uint32)
#         diff = abs(int(l[0]) - length)
#         if len(w) > length and not np.array_equal(data1, np.zeros((0,0))):
#             data1 = _add_empty(data1, diff)
#         elif word.shape[0] < length and not np.array_equal(data1, np.zeros((0,0))):
#             word = _add_empty(word, diff)
#
#         if np.array_equal(data1, np.zeros((0,0))):
#             data1 = word
#         else:
#             data1 = np.concatenate((data1, word), axis=0)
#         length = data1.shape[1]
#         line = f.readline()
#
#     f.close()
#     if pickle:
#         _create_pickle_files(adr=adr, dsample=dsample)
#     return nbL, nbEx, data1


# def _add_empty(data, diff):
#     empty = np.zeros((data.shape[0], diff))
#     empty += -1
#     data = np.concatenate((data, empty), axis=1)
#     return data


def _create_pickle_files(self, adr, dsample):
    f = open(adr + ".sample.pkl", "wb")
    pickle.dump(dsample, f)
    f.close()
