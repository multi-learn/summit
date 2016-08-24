from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


def fit(DATASET, CLASS_LABELS, NB_CORES=1,**kwargs):
    num_estimators = int(kwargs['0'])
    base_estimators = int(kwargs['1'])
    classifier = AdaBoostClassifier(n_estimators=num_estimators, base_estimator=base_estimators)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def fit_gridsearch(X_train, y_train, nbFolds=4, nbCores=1, **kwargs):

    pipeline = Pipeline([('classifier', AdaBoostClassifier())])
    param= {"classifier__n_estimators": map(int, kwargs['0']),
                "classifier__base_estimator": [DecisionTreeClassifier() for arg in kwargs["1"]]}
    grid = GridSearchCV(pipeline,param_grid=param,refit=True,n_jobs=nbCores,scoring='accuracy',cv=nbFolds)
    detector = grid.fit(X_train, y_train)
    desc_estimators = [detector.best_params_["classifier__n_estimators"]]
    description = "Classif_" + "RF" + "-" + "CV_" +  str(nbFolds) + "-" + "Trees_" + str(map(str,desc_estimators))
    return description, detector


def getConfig(config):
    return "\n\t\t- Adaboost with num_esimators : "+config[0]+", base_estimators : "+config[1]