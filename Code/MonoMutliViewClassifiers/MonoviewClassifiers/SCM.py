from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
import Metrics
from pyscm.utils import _pack_binary_bytes_to_ints
import pyscm
from scipy.stats import randint
from utils.Dataset import getShape
import h5py
from pyscm.binary_attributes.base import BaseBinaryAttributeList
import logging
# Author-Info
__author__ 	= "Baptiste Bauvin"
__status__ 	= "Prototype"                           # Production, Development, Prototype


def fit(DATASET, CLASS_LABELS, NB_CORES=1,**kwargs):
    try:
        logging.debug("Start:\t Getting pre-computed rules")
        attributeClassification = kwargs["attributeClassification"]
        binaryAttributes = kwargs["binaryAttributes"]
        logging.debug("Done:\t Getting pre-computed rules")
    except:
        logging.debug("Start:\t Pre-computing rules")
        attributeClassification, binaryAttributes = transformData(DATASET)
        logging.debug("Done:\t Pre-computing rules")
    classifier = pyscm.scm.SetCoveringMachine(p=1.0, max_attributes=10, verbose=False)
    classifier.fit(binaryAttributes, CLASS_LABELS, X=None, attribute_classifications=attributeClassification, iteration_callback=None)
    return classifier


def gridSearch(X_train, y_train, nbFolds=4, metric=["accuracy_score", None], nIter=30, nbCores=1):

    # pipeline = Pipeline([('classifier',  pyscm.scm.SetCoveringMachine())])
    #
    # param= {"classifier__max_attributes": randint(1, 15),
    #         "classifier__c":[1.0] ,
    #         "classifier__p":[1.0] }
    # metricModule = getattr(Metrics, metric[0])
    # if metric[1]!=None:
    #     metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    # else:
    #     metricKWARGS = {}
    # scorer = metricModule.get_scorer(**metricKWARGS)
    # grid = RandomizedSearchCV(pipeline, n_iter=nIter, param_distributions=param,refit=True,n_jobs=nbCores,scoring=scorer,cv=nbFolds)
    # detector = grid.fit(X_train, y_train)
    desc_estimators = [5]
    return desc_estimators


def getConfig(config):
    try :
        return "\n\t\t- SCM with max_attributes : "+str(config[0])#+", c : "+str(config[1])+", p : "+str(config[2])
    except:
        return "\n\t\t- SCM with max_attributes : "+str(config["0"])#+", c : "+str(config["1"])+", p : "+str(config["2"])


def transformData(dataArray):
    dataArray = dataArray.astype(np.uint8)
    if isBinary(dataArray):
        nbExamples = dataArray.shape[0]
        featureSequence = [str(featureIndex) for featureIndex in range(dataArray.shape[1])]
        featureIndexByRule = np.arange(dataArray.shape[1], dtype=np.uint32)
        logging.debug("Start:\t Creating binary attributes")
        binaryAttributes = LazyBaptisteRuleList(featureSequence, featureIndexByRule)
        logging.debug("Done:\t Creating binary attributes")
        logging.debug("Start:\t Packing Data")
        packedData = _pack_binary_bytes_to_ints(dataArray, 64)
        del dataArray
        dsetFile = h5py.File("temp_scm", "w")
        packedDataset = dsetFile.create_dataset("temp_scm", data=packedData)
        dsetFile.close()
        packedDataset = h5py.File("temp_scm", "r").get("temp_scm")
        logging.debug("Done:\t Packing Data")
        attributeClassification = BaptisteRuleClassifications(packedDataset, nbExamples)
        return attributeClassification, binaryAttributes


def isBinary(dataset):
    if type(dataset[0,0]) is np.uint8:
        return True
    for line in dataset:
        for data in line:
            if data!=0 or data!=1:
                return False
    return True

#!/usr/bin/env python
"""
	Kover: Learn interpretable computational phenotyping models from k-merized genomic data
	Copyright (C) 2015  Alexandre Drouin
	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.
	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.
	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

from math import ceil

from pyscm.binary_attributes.classifications.popcount import inplace_popcount_32, inplace_popcount_64
from pyscm.utils import _unpack_binary_bytes_from_ints


def _minimum_uint_size(max_value):
    """
    Find the minimum size unsigned integer type that can store values of at most max_value
    From A.Drouin's Kover
    """
    if max_value <= np.iinfo(np.uint8).max:
        return np.uint8
    elif max_value <= np.iinfo(np.uint16).max:
        return np.uint16
    elif max_value <= np.iinfo(np.uint32).max:
        return np.uint32
    elif max_value <= np.iinfo(np.uint64).max:
        return np.uint64
    else:
        return np.uint128


class BaptisteRule(object):

    def __init__(self, feature_index, kmer_sequence, type):
        """
        A k-mer rule
        Parameters:
        -----------
        feature_index: uint
            The index of the k-mer
        kmer_sequence: string
            The nucleotide sequence of the k-mer
        type: string
            The type of rule: presence or absence (use p or a)
        """
        self.feature_index = feature_index
        self.kmer_sequence = kmer_sequence
        self.type = type

    def classify(self, X):
        if self.type == "absence":
            return (X[:, self.feature_index] == 0).astype(np.uint8)
        else:
            return (X[:, self.feature_index] == 1).astype(np.uint8)

    def inverse(self):
        return BaptisteRule(feature_index=self.feature_index, kmer_sequence=self.kmer_sequence, type="absence" if self.type == "presence" else "presence")

    def __str__(self):
        return ("Absence(" if self.type == "absence" else "Presence(") + self.kmer_sequence + ")"

class LazyBaptisteRuleList(object):
    """
    By convention, the first half of the list contains presence rules and the second half contains the absence rules in
    the same order.
    """
    def __init__(self, kmer_sequences, feature_index_by_rule):
        self.n_rules = feature_index_by_rule.shape[0] * 2
        self.kmer_sequences = kmer_sequences
        self.feature_index_by_rule = feature_index_by_rule
        super(LazyBaptisteRuleList, self).__init__()

    def __getitem__(self, idx):
        if idx >= self.n_rules:
            raise ValueError("Index %d is out of range for list of size %d" % (idx, self.n_rules))
        if idx >= len(self.kmer_sequences):
            type = "absence"
            feature_idx = self.feature_index_by_rule[idx % len(self.kmer_sequences)]
        else:
            type = "presence"
            feature_idx = self.feature_index_by_rule[idx]
        return BaptisteRule(idx % len(self.kmer_sequences), self.kmer_sequences[feature_idx], type)

    def __len__(self):
        return self.n_rules

class BaseRuleClassifications(object):
    def __init__(self):
        pass

    def get_columns(self, columns):
        raise NotImplementedError()

    def remove_rows(self, rows):
        raise NotImplementedError()

    @property
    def shape(self):
        raise NotImplementedError()

    def sum_rows(self, rows):
        raise NotImplementedError()


class BaptisteRuleClassifications(BaseRuleClassifications):
    """
    Methods involving columns account for presence and absence rules
    """
    # TODO: Clean up. Get rid of the code to handle deleted rows. We don't need this.
    def __init__(self, dataset, n_rows, block_size=None):
        self.dataset = dataset
        self.dataset_initial_n_rows = n_rows
        self.dataset_n_rows = n_rows
        self.dataset_removed_rows = []
        self.dataset_removed_rows_mask = np.zeros(self.dataset_initial_n_rows, dtype=np.bool)
        self.block_size = (None, None)

        if block_size is None:
            if self.dataset.chunks is None:
                self.block_size = (1, self.dataset.shape[1])
            else:
                self.block_size = self.dataset.chunks
        else:
            if len(block_size) != 2 or not isinstance(block_size[0], int) or not isinstance(block_size[1], int):
                raise ValueError("The block size must be a tuple of 2 integers.")
            self.block_size = block_size

        # Get the size of the ints used to store the data
        if self.dataset.dtype == np.uint32:
            self.dataset_pack_size = 32
            self.inplace_popcount = inplace_popcount_32
        elif self.dataset.dtype == np.uint64:
            self.dataset_pack_size = 64
            self.inplace_popcount = inplace_popcount_64
        else:
            raise ValueError("Unsupported data type for packed attribute classifications array. The supported data" +
                             " types are np.uint32 and np.uint64.")

        super(BaseRuleClassifications, self).__init__()

    def get_columns(self, columns):
        """
        Columns can be an integer (or any object that implements __index__) or a sorted list/ndarray.
        """
        #TODO: Support slicing, make this more efficient than getting the columns individually.
        columns_is_int = False
        if hasattr(columns, "__index__"):  # All int types implement the __index__ method (PEP 357)
            columns = [columns.__index__()]
            columns_is_int = True
        elif isinstance(columns, np.ndarray):
            columns = columns.tolist()
        elif isinstance(columns, list):
            pass
        else:
            columns = list(columns)
        # Detect where an inversion is needed (columns corresponding to absence rules)
        columns, invert_result = zip(* (((column if column < self.dataset.shape[1] else column % self.dataset.shape[1]),
                                         (True if column > self.dataset.shape[1] else False)) for column in columns))
        columns = list(columns)
        invert_result = np.array(invert_result)

        # Don't return rows that have been deleted
        row_mask = np.ones(self.dataset.shape[0] * self.dataset_pack_size, dtype=np.bool)
        row_mask[self.dataset_initial_n_rows:] = False
        row_mask[self.dataset_removed_rows] = False

        # h5py requires that the column indices are sorted
        unique, inverse = np.unique(columns, return_inverse=True)
        result = _unpack_binary_bytes_from_ints(self.dataset[:, unique.tolist()])[row_mask]
        result = result[:, inverse]
        result[:, invert_result] = 1 - result[:, invert_result]

        if columns_is_int:
            return result.reshape(-1)
        else:
            return result

    @property
    def shape(self):
        return self.dataset_n_rows, self.dataset.shape[1] * 2

    # TODO: allow summing over multiple lists of rows at a time (saves i/o operations)
    def sum_rows(self, rows):
        """
        Note: Assumes that the rows argument does not contain duplicate elements. Rows will not be considered more than once.
        """
        rows = np.asarray(rows)
        result_dtype = _minimum_uint_size(rows.shape[0])
        result = np.zeros(self.dataset.shape[1] * 2, dtype=result_dtype)

        # Builds a mask to turn off the bits of the rows we do not want to count in the sum.
        def build_row_mask(example_idx, n_examples, mask_n_bits):
            if mask_n_bits not in [8, 16, 32, 64, 128]:
                raise ValueError("Unsupported mask format. Use 8, 16, 32, 64 or 128 bits.")

            n_masks = int(ceil(float(n_examples) / mask_n_bits))
            masks = [0] * n_masks

            for idx in example_idx:
                example_mask = idx / mask_n_bits
                example_mask_idx = mask_n_bits - (idx - mask_n_bits * example_mask) - 1
                masks[example_mask] |= 1 << example_mask_idx

            return np.array(masks, dtype="u" + str(mask_n_bits / 8))

        # Find the rows that occur in each dataset and their relative index
        rows = np.sort(rows)
        dataset_relative_rows = []
        for row_idx in rows:
            # Find which row in the dataset corresponds to the requested row
            # TODO: This is inefficient! Could exploit the fact that rows is sorted to reuse previous iterations.
            current_idx = -1
            n_active_elements_seen = 0
            while n_active_elements_seen <= row_idx:
                current_idx += 1
                if not self.dataset_removed_rows_mask[current_idx]:
                    n_active_elements_seen += 1
            dataset_relative_rows.append(current_idx)

        # Create a row mask for each dataset
        row_mask = build_row_mask(dataset_relative_rows, self.dataset_initial_n_rows, self.dataset_pack_size)
        del dataset_relative_rows

        # For each dataset load the rows for which the mask is not 0. Support column slicing aswell
        n_col_blocks = int(ceil(1.0 * self.dataset.shape[1] / self.block_size[1]))
        rows_to_load = np.where(row_mask != 0)[0]
        n_row_blocks = int(ceil(1.0 * len(rows_to_load) / self.block_size[0]))

        for row_block in xrange(n_row_blocks):
            block_row_mask = row_mask[rows_to_load[row_block * self.block_size[0]:(row_block + 1) * self.block_size[0]]]

            for col_block in xrange(n_col_blocks):

                # Load the appropriate rows/columns based on the block sizes
                block = self.dataset[rows_to_load[row_block * self.block_size[0]:(row_block + 1) * self.block_size[0]],
                        col_block * self.block_size[1]:(col_block + 1) * self.block_size[1]]

                # Popcount
                if len(block.shape) == 1:
                    block = block.reshape(1, -1)
                self.inplace_popcount(block, block_row_mask)

                # Increment the sum
                result[col_block * self.block_size[1]:min((col_block + 1) * self.block_size[1], self.dataset.shape[1])] += np.sum(block, axis=0)

        # Compute the sum for absence rules
        result[self.dataset.shape[1] : ] = len(rows) - result[: self.dataset.shape[1]]

        return result