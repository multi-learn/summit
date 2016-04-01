import numpy as np

def initialize(NB_CLASS, NB_VIEW, NB_ITER, DATASET_LENGTH, CLASS_LABELS):
    costMatrices = np.array([
                    np.array([
                        np.array([
                            np.array([ 1 if CLASS_LABELS[exampleIndice]!=classe
                                       else -(NB_CLASS-1)
                                       for classe in range(NB_CLASS)
                                    ]) for exampleIndice in range(DATASET_LENGTH)
                            ]) for viewIndice in range(NB_VIEW)]) for iteration in range(NB_ITER)
                    ])
    return costMatrices, generalCostMatrix, f