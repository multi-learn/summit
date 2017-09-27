from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import Metrics
from scipy.stats import randint


# Author-Info
__author__ 	= "Baptiste Bauvin"
__status__ 	= "Prototype"                           # Production, Development, Prototype


def canProbas():
    return True


def fit(DATASET, CLASS_LABELS, NB_CORES=1,**kwargs):
    num_estimators = int(kwargs['0'])
    maxDepth = int(kwargs['1'])
    classifier = RandomForestClassifier(n_estimators=num_estimators, max_depth=maxDepth, n_jobs=NB_CORES)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def gridSearch(X_train, y_train, nbFolds=4, nbCores=1, metric=["accuracy_score", None], nIter=30):
    pipeline_rf = Pipeline([('classifier', RandomForestClassifier())])
    param_rf = {"classifier__n_estimators": randint(1, 30),
                "classifier__max_depth":randint(1, 30)}
    metricModule = getattr(Metrics, metric[0])
    if metric[1]!=None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}
    scorer = metricModule.get_scorer(**metricKWARGS)
    grid_rf = RandomizedSearchCV(pipeline_rf, n_iter=nIter,param_distributions=param_rf,refit=True,n_jobs=nbCores,scoring=scorer,cv=nbFolds)
    rf_detector = grid_rf.fit(X_train, y_train)

    desc_estimators = [rf_detector.best_params_["classifier__n_estimators"],
                       rf_detector.best_params_["classifier__max_depth"]]
    return desc_estimators


def getConfig(config):
    try:
        return "\n\t\t- Random Forest with num_esimators : "+str(config[0])+", max_depth : "+str(config[1])
    except:
        return "\n\t\t- Random Forest with num_esimators : "+str(config["0"])+", max_depth : "+str(config["1"])
