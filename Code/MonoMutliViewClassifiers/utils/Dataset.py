import h5py
from scipy import sparse


def getV(DATASET, viewIndex, usedIndices=None):
    if usedIndices==None:
        usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
    if not DATASET.get("View"+str(viewIndex)).attrs["sparse"]:
        return DATASET.get("View"+str(viewIndex))[usedIndices, :]
    else:
        return sparse.csr_matrix((DATASET.get("View"+str(viewIndex)).get("data").value,
                                  DATASET.get("View"+str(viewIndex)).get("indices").value,
                                  DATASET.get("View"+str(viewIndex)).get("indptr").value),
                                 shape=DATASET.get("View"+str(viewIndex)).attrs["shape"])[usedIndices,:]


def getShape(DATASET, viewIndex):
    if not DATASET.get("View"+str(viewIndex)).attrs["sparse"]:
        return DATASET.get("View"+str(viewIndex)).shape
    else:
        return DATASET.get("View"+str(viewIndex)).attrs["shape"]


def getValue(DATASET):
    if not DATASET.attrs["sparse"]:
        return DATASET.value
    else:
        return sparse.csr_matrix((DATASET.get("data").value,
                                  DATASET.get("indices").value,
                                  DATASET.get("indptr").value),
                                 shape=DATASET.attrs["shape"])