from scipy import sparse
import numpy as np


def getV(DATASET, viewIndex, usedIndices=None):
    if usedIndices==None:
        usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
    if not DATASET.get("View"+str(viewIndex)).attrs["sparse"]:
        return DATASET.get("View"+str(viewIndex))[usedIndices, :]
    else:
        sparse_mat = sparse.csr_matrix((DATASET.get("View"+str(viewIndex)).get("data").value,
                                       DATASET.get("View"+str(viewIndex)).get("indices").value,
                                       DATASET.get("View"+str(viewIndex)).get("indptr").value),
                                      shape=DATASET.get("View"+str(viewIndex)).attrs["shape"])[usedIndices,:]
        print sparse_mat.shape
        print sparse_mat.indptr
        return sparse_mat


def getShape(DATASET, viewIndex):
    if not DATASET.get("View"+str(viewIndex)).attrs["sparse"]:
        return DATASET.get("View"+str(viewIndex)).shape
    else:
        return DATASET.get("View"+str(viewIndex)).attrs["shape"]


def getValue(DATASET):
    if not DATASET.attrs["sparse"]:
        return DATASET.value
    else:
        sparse_mat = sparse.csr_matrix((DATASET.get("data").value,
                                  DATASET.get("indices").value,
                                  DATASET.get("indptr").value),
                                 shape=DATASET.attrs["shape"])
        print sparse_mat.shape
        print sparse_mat.indptr
        return sparse_mat

def extractSubset(matrix, usedIndices):
    if sparse.issparse(matrix):
        newIndptr = np.zeros(len(usedIndices)+1, dtype=np.int16)
        oldindptr = matrix.indptr
        print oldindptr
        for exampleIndexIndex, exampleIndex in enumerate(usedIndices):
            newIndptr[exampleIndexIndex+1] = newIndptr[exampleIndexIndex]+(oldindptr[exampleIndex+1]-oldindptr[exampleIndex])
        newData = np.ones(newIndptr[-1], dtype=bool)
        newIndices =  np.zeros(newIndptr[-1], dtype=np.int32)
        oldIndices = matrix.indices
        print newIndptr
        for exampleIndexIndex, exampleIndex in enumerate(usedIndices):
            print newIndptr[exampleIndexIndex], newIndptr[exampleIndexIndex+1]
            newIndices[newIndptr[exampleIndexIndex]:newIndptr[exampleIndexIndex+1]] = oldIndices[oldindptr[exampleIndex]: oldindptr[exampleIndex+1]]
        return sparse.csr_matrix((newData, newIndices, newIndptr), shape=(len(usedIndices), matrix.shape))
    else:
        return matrix[usedIndices]