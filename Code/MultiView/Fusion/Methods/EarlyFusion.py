#!/usr/bin/env python
# -*- encoding: utf-8

from MonoviewClassifiers import *

import numpy as np

def makemonoviewData(DATASET, weights):
    NB_VIEW = len(DATASET)
    DATASET_LENGTH = len(DATASET[0])
    weightedMonoviewData = np.array([np.concatenate([weights[viewIndex]*DATASET[viewIndex][exampleIndex]
                                                     for viewIndex in np.arange(NB_VIEW)])
                                     for exampleIndex in np.arange(DATASET_LENGTH)])
    return  weightedMonoviewData

def linearWeightedTrain(DATASET, CLASS_LABELS, DATASET_LENGTH, NB_VIEW, monoviewClassifier,
                   monoviewClassifierConfig, fusionConfig):
    weights = map(float, fusionConfig)
    weightedMonoviewData = makemonoviewData(DATASET, weights)
    monoviewClassifierModule = globals()[monoviewClassifier]
    classifier = monoviewClassifierModule.train(weightedMonoviewData, CLASS_LABELS, monoviewClassifierConfig)
    return (classifier, weights)


def linearWeightedPredict(DATASET, fusionClassifier):
    classifier, weights = fusionClassifier
    weightedMonoviewData = makemonoviewData(DATASET, weights)
    predictedLabels = classifier.predict(weightedMonoviewData)
    return predictedLabels

def getConfig(fusionMethodConfig, monoviewClassifier, fusionClassifierConfig):
    return