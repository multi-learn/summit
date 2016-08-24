from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline                   # Pipelining in classification
from sklearn.grid_search import GridSearchCV


def fit(DATASET, CLASS_LABELS, NB_CORES=1,**kwargs):
    loss = kwargs['0']
    penalty = kwargs['1']
    try:
        alpha = int(kwargs['2'])
    except:
        alpha = 0.15
    classifier = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def fit_gridsearch(X_train, y_train, nbFolds=4, nbCores=1, **kwargs):
    pipeline_SGD = Pipeline([('classifier', SGDClassifier())])
    param_SGD = {"classifier__loss": kwargs['1'], "classifier__penalty": kwargs['2'],
                 "classifier__alpha": map(float, kwargs['0'])}
    grid_SGD = GridSearchCV(pipeline_SGD, param_grid=param_SGD, refit=True, n_jobs=nbCores, scoring='accuracy',
                            cv=nbFolds)
    SGD_detector = grid_SGD.fit(X_train, y_train)
    desc_params = [SGD_detector.best_params_["classifier__loss"], SGD_detector.best_params_["classifier__penalty"],
                   SGD_detector.best_params_["classifier__alpha"]]
    description = "Classif_" + "Lasso" + "-" + "CV_" + str(nbFolds) + "-" + "-".join(map(str,desc_params))
    return description, SGD_detector


def getConfig(config):
    return "\n\t\t- SGDClassifier with loss : "+config[0]+", penalty : "+config[1]