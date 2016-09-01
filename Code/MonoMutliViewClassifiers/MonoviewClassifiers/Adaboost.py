from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
import Metrics
from scipy.stats import randint

def fit(DATASET, CLASS_LABELS, NB_CORES=1,**kwargs):
    num_estimators = int(kwargs['0'])
    base_estimators = DecisionTreeClassifier()#kwargs['1']
    classifier = AdaBoostClassifier(n_estimators=num_estimators, base_estimator=base_estimators)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier

#
# def fit_gridsearch(X_train, y_train, nbFolds=4, nbCores=1, metric=["accuracy_score", None], **kwargs):
#     pipeline = Pipeline([('classifier', AdaBoostClassifier())])
#     metricModule = getattr(Metrics, metric[0])
#     scorer = metricModule.get_scorer(dict((index, metricConfig) for index, metricConfig in enumerate(metric[1])))
#     param= {"classifier__n_estimators": map(int, kwargs['0']),
#                 "classifier__base_estimator": [DecisionTreeClassifier() for arg in kwargs["1"]]}
#     grid = RandomizedSearchCV(pipeline,param_distributions=param,refit=True,n_jobs=nbCores,scoring=scorer,cv=nbFolds)
#     detector = grid.fit(X_train, y_train)
#     desc_estimators = [detector.best_params_["classifier__n_estimators"]]
#     description = "Classif_" + "RF" + "-" + "CV_" +  str(nbFolds) + "-" + "Trees_" + str(map(str,desc_estimators))
#     return description, detector


def gridSearch(X_train, y_train, nbFolds=4, metric=["accuracy_score", None], nIter=30, nbCores=1):

    pipeline = Pipeline([('classifier', AdaBoostClassifier())])
    # classifiers = [clf for name, clf in all_estimators(type_filter='classifier')
    #                if 'sample_weight' in inspect.getargspec(clf().fit)[0]
    #                and (name != "AdaBoostClassifier" and name !="GradientBoostingClassifier" )]

    param= {"classifier__n_estimators": randint(1, 15),
            "classifier__base_estimator": [DecisionTreeClassifier()]}
    metricModule = getattr(Metrics, metric[0])
    if metric[1]!=None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}
    scorer = metricModule.get_scorer(**metricKWARGS)
    grid = RandomizedSearchCV(pipeline, n_iter=nIter, param_distributions=param,refit=True,n_jobs=nbCores,scoring=scorer,cv=nbFolds)
    detector = grid.fit(X_train, y_train)
    desc_estimators = [detector.best_params_["classifier__n_estimators"],
                       detector.best_params_["classifier__base_estimator"]]
    return desc_estimators


def getConfig(config):
    try :
        return "\n\t\t- Adaboost with num_esimators : "+str(config[0])+", base_estimators : "+str(config[1])
    except:
        return "\n\t\t- Adaboost with num_esimators : "+str(config["0"])+", base_estimators : "+str(config["1"])