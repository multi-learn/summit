from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
import Metrics
from scipy.stats import randint

# Author-Info
__author__ 	= "Baptiste Bauvin"
__status__ 	= "Prototype"                           # Production, Development, Prototype


def canProbas():
    return True

def fit(DATASET, CLASS_LABELS, NB_CORES=1,**kwargs):
    num_estimators = int(kwargs['0'])
    base_estimators = DecisionTreeClassifier()#kwargs['1']
    classifier = AdaBoostClassifier(n_estimators=num_estimators, base_estimator=base_estimators)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def gridSearch(X_train, y_train, nbFolds=4, metric=["accuracy_score", None], nIter=30, nbCores=1):

    pipeline = Pipeline([('classifier', AdaBoostClassifier())])

    param= {"classifier__n_estimators": randint(1, 15),
            "classifier__base_estimator": [DecisionTreeClassifier()]}
    metricModule = getattr(Metrics, metric[0])
    if metric[1]!=None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}
    scorer = metricModule.get_scorer(**metricKWARGS)
    grid = RandomizedSearchCV(pipeline, n_iter=nIter, param_distributions=param, refit=True, n_jobs=nbCores, scoring=scorer, cv=nbFolds)
    detector = grid.fit(X_train, y_train)
    desc_estimators = [detector.best_params_["classifier__n_estimators"],
                       detector.best_params_["classifier__base_estimator"]]
    return desc_estimators


def getConfig(config):
    try :
        return "\n\t\t- Adaboost with num_esimators : "+str(config[0])+", base_estimators : "+str(config[1])
    except:
        return "\n\t\t- Adaboost with num_esimators : "+str(config["0"])+", base_estimators : "+str(config["1"])