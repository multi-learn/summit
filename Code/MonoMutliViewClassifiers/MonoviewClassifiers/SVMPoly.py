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
    degree = int(kwargs['1'])
    classifier = SVC(C=C, kernel='poly', degree=degree, probability=True, max_iter=1000, random_state=randomState)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append([randomState.randint(1, 10000), randomState.randint(1, 30)])
    return paramsSet


def getKWARGS(kwargsList):
    kwargsDict = {}
    for (kwargName, kwargValue) in kwargsList:
        if kwargName == "CL_SVMPoly_C":
            kwargsDict['0'] = int(kwargValue)
        elif kwargName == "CL_SVMPoly_deg":
            kwargsDict['1'] = int(kwargValue)
    return kwargsDict


def randomizedSearch(X_train, y_train, randomState, KFolds=4, nbCores=1, metric=["accuracy_score", None], nIter=30):
    pipeline_SVMPoly = Pipeline([('classifier', SVC(kernel="poly", max_iter=1000))])
    param_SVMPoly = {"classifier__C": randint(1, 10000),
                     "classifier__degree": randint(1, 30)}
    metricModule = getattr(Metrics, metric[0])
    if metric[1]!=None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}
    scorer = metricModule.get_scorer(**metricKWARGS)
    grid_SVMPoly = RandomizedSearchCV(pipeline_SVMPoly, n_iter=nIter, param_distributions=param_SVMPoly, refit=True,
                                      n_jobs=nbCores, scoring=scorer, cv=KFolds, random_state=randomState)
    SVMRBF_detector = grid_SVMPoly.fit(X_train, y_train)
    desc_params = [SVMRBF_detector.best_params_["classifier__C"], SVMRBF_detector.best_params_["classifier__degree"]]
    return desc_params


def getConfig(config):
    if type(config) not in [list, dict]:
        return "\n\t\t- SVM Poly with C : "+str(config.C)+", degree : "+str(config.degree)
    else:
        try:
            return "\n\t\t- SVM Poly with C : "+str(config[0])+", degree : "+str(config[1])
        except:
            return "\n\t\t- SVM Poly with C : "+str(config["0"])+", degree : "+str(config["1"])