from sklearn.svm import SVC
from sklearn.pipeline import Pipeline                   # Pipelining in classification
from sklearn.grid_search import RandomizedSearchCV
import Metrics
from scipy.stats import randint


# Author-Info
__author__ 	= "Baptiste Bauvin"
__status__ 	= "Prototype"                           # Production, Development, Prototype


def canProbas():
    return True

def fit(DATASET, CLASS_LABELS, NB_CORES=1,**kwargs):
    C = int(kwargs['0'])
    classifier = SVC(C=C, kernel='linear', probability=True, max_iter=1000)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def gridSearch(X_train, y_train, nbFolds=4, nbCores=1, metric=["accuracy_score", None], nIter=30):
    pipeline_SVMLinear = Pipeline([('classifier', SVC(kernel="linear", max_iter=1000))])
    param_SVMLinear = {"classifier__C":randint(1, 10000)}
    metricModule = getattr(Metrics, metric[0])
    if metric[1]!=None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}
    scorer = metricModule.get_scorer(**metricKWARGS)
    grid_SVMLinear = RandomizedSearchCV(pipeline_SVMLinear, n_iter=nIter,param_distributions=param_SVMLinear, refit=True, n_jobs=nbCores, scoring=scorer,
                                  cv=nbFolds)

    SVMLinear_detector = grid_SVMLinear.fit(X_train, y_train)
    desc_params = [SVMLinear_detector.best_params_["classifier__C"]]
    return desc_params


def getConfig(config):
    try:
        return "\n\t\t- SVM Linear with C : "+str(config[0])
    except:
        return "\n\t\t- SVM Linear with C : "+str(config["0"])
