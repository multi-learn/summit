from sklearn.svm import SVC
from sklearn.pipeline import Pipeline                   # Pipelining in classification
from sklearn.model_selection import RandomizedSearchCV
import Metrics
from scipy.stats import randint


# Author-Info
__author__ 	= "Baptiste Bauvin"
__status__ 	= "Prototype"                           # Production, Development, Prototype


def canProbas():
    return True


def fit(DATASET, CLASS_LABELS, randomState, NB_CORES=1,**kwargs):
    C = int(kwargs['0'])
    classifier = SVC(C=C, kernel='rbf', probability=True, max_iter=1000, random_state=randomState)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def getKWARGS(kwargsList):
    kwargsDict = {}
    for (kwargName, kwargValue) in kwargsList:
        if kwargName == "CL_SVMRBF_C":
            kwargsDict['0'] = int(kwargValue)
    return kwargsDict


def randomizedSearch(X_train, y_train, randomState, nbFolds=4, nbCores=1, metric=["accuracy_score", None], nIter=30):
    pipeline_SVMRBF = Pipeline([('classifier', SVC(kernel="rbf", max_iter=1000))])
    param_SVMRBF = {"classifier__C": randint(1, 10000)}
    metricModule = getattr(Metrics, metric[0])
    if metric[1]!=None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}
    scorer = metricModule.get_scorer(**metricKWARGS)
    grid_SVMRBF = RandomizedSearchCV(pipeline_SVMRBF, n_iter=nIter, param_distributions=param_SVMRBF, refit=True,
                                     n_jobs=nbCores, scoring=scorer, cv=nbFolds, random_state=randomState)
    SVMRBF_detector = grid_SVMRBF.fit(X_train, y_train)
    desc_params = [SVMRBF_detector.best_params_["classifier__C"]]
    return desc_params


def getConfig(config):
    try:
        return "\n\t\t- SVM RBF with C : "+str(config[0])
    except:
        return "\n\t\t- SVM RBF with C : "+str(config["0"])