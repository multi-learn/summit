from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline                   # Pipelining in classification
from sklearn.model_selection import RandomizedSearchCV
import Metrics
from scipy.stats import randint


# Author-Info
__author__ 	= "Baptiste Bauvin"
__status__ 	= "Prototype"                           # Production, Development, Prototype


def canProbas():
    return True

def fit(DATASET, CLASS_LABELS, NB_CORES=1,**kwargs):
    nNeighbors = int(kwargs['0'])
    classifier = KNeighborsClassifier(n_neighbors=nNeighbors)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def gridSearch(X_train, y_train, nbFolds=4, nbCores=1, metric=["accuracy_score", None], nIter=30 ):
    pipeline_KNN = Pipeline([('classifier', KNeighborsClassifier())])
    param_KNN = {"classifier__n_neighbors": randint(1, 50)}
    metricModule = getattr(Metrics, metric[0])
    if metric[1]!=None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}
    scorer = metricModule.get_scorer(**metricKWARGS)
    grid_KNN = RandomizedSearchCV(pipeline_KNN, n_iter=nIter, param_distributions=param_KNN, refit=True, n_jobs=nbCores, scoring=scorer,
                            cv=nbFolds)
    KNN_detector = grid_KNN.fit(X_train, y_train)
    desc_params = [KNN_detector.best_params_["classifier__n_neighbors"]]
    return desc_params


def getConfig(config):
    try:
        return "\n\t\t- K nearest Neighbors with  n_neighbors: "+str(config[0])
    except:
        return "\n\t\t- K nearest Neighbors with  n_neighbors: "+str(config["0"])