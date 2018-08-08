from sklearn import tree
from sklearn.pipeline import Pipeline  # Pipelining in classification
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import numpy as np
# import graphviz
# import cPickle

from .. import Metrics
from ..utils.HyperParameterSearch import genHeatMaps

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def canProbas():
    return True


def fit(DATASET, CLASS_LABELS, randomState, NB_CORES=1, **kwargs):
    classifier = tree.DecisionTreeClassifier(max_depth=kwargs['max_depth'], criterion=kwargs['criterion'],
                                             splitter=kwargs['splitter'], random_state=randomState)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"max_depth": randomState.randint(1, 300),
                          "criterion": randomState.choice(["gini", "entropy"]),
                          "splitter": randomState.choice(["best", "random"])})
    return paramsSet


def getKWARGS(args):
    kwargsDict = {"max_depth": args.DT_depth, "criterion": args.DT_criterion, "splitter": args.DT_splitter}
    return kwargsDict


def genPipeline():
    return Pipeline([('classifier', tree.DecisionTreeClassifier())])


def genParamsDict(randomState):
    return {"classifier__max_depth": np.arange(1, 300),
                "classifier__criterion": ["gini", "entropy"],
                "classifier__splitter": ["best", "random"]}


def genBestParams(detector):
    return {"max_depth": detector.best_params_["classifier__max_depth"],
            "criterion": detector.best_params_["classifier__criterion"],
            "splitter": detector.best_params_["classifier__splitter"]}


def genParamsFromDetector(detector):
    return [("maxDepth", np.array(detector.cv_results_['param_classifier__max_depth'])),
            ("criterion", np.array(detector.cv_results_['param_classifier__criterion'])),
            ("splitter", np.array(detector.cv_results_['param_classifier__splitter']))]


def getConfig(config):
    if type(config) is not dict:
        return "\n\t\t- Decision Tree with max_depth : " + str(
            config.max_depth) + ", criterion : " + config.criterion + ", splitter : " + config.splitter
    else:
        return "\n\t\t- Decision Tree with max_depth : " + str(config["max_depth"]) + ", criterion : " + config[
                "criterion"] + ", splitter : " + config["splitter"]

def getInterpret(classifier, directory):
    dot_data = tree.export_graphviz(classifier, out_file=None)
    # graph = graphviz.Source(dot_data)
    # graph.render(directory+"-tree.pdf")
    interpretString = getFeatureImportance(classifier, directory)
    return interpretString