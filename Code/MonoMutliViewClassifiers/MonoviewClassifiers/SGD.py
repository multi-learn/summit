from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline                   # Pipelining in classification
from sklearn.grid_search import GridSearchCV
import numpy as np
import Metrics


def fit(DATASET, CLASS_LABELS, NB_CORES=1,**kwargs):
    loss = kwargs['0']
    penalty = kwargs['1']
    try:
        alpha = int(kwargs['2'])
    except:
        alpha = 0.15
    classifier = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha)
    classifier.fit(DATASET, CLASS_LABELS)
    return "No desc", classifier


# def fit_gridsearch(X_train, y_train, nbFolds=4, nbCores=1, metric=["accuracy_score", None], **kwargs):
#     pipeline_SGD = Pipeline([('classifier', SGDClassifier())])
#     param_SGD = {"classifier__loss": kwargs['1'], "classifier__penalty": kwargs['2'],
#                  "classifier__alpha": map(float, kwargs['0'])}
#     metricModule = getattr(Metrics, metric[0])
#     scorer = metricModule.get_scorer(dict((index, metricConfig) for index, metricConfig in enumerate(metric[1])))
#     grid_SGD = GridSearchCV(pipeline_SGD, param_grid=param_SGD, refit=True, n_jobs=nbCores, scoring='accuracy',
#                             cv=nbFolds)
#     SGD_detector = grid_SGD.fit(X_train, y_train)
#     desc_params = [SGD_detector.best_params_["classifier__loss"], SGD_detector.best_params_["classifier__penalty"],
#                    SGD_detector.best_params_["classifier__alpha"]]
#     description = "Classif_" + "Lasso" + "-" + "CV_" + str(nbFolds) + "-" + "-".join(map(str,desc_params))
#     return description, SGD_detector


def gridSearch(X_train, y_train, nbFolds=4, nbCores=1, metric=["accuracy_score", None], **kwargs):
    pipeline_SGD = Pipeline([('classifier', SGDClassifier())])
    losses = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']
    penalties = ["l1", "l2", "elasticnet"]
    alphas = list(np.random.randint(1,10,10))+list(np.random.random_sample(10))
    param_SGD = {"classifier__loss": losses, "classifier__penalty": penalties,
                 "classifier__alpha": alphas}
    metricModule = getattr(Metrics, metric[0])
    scorer = metricModule.get_scorer(dict((index, metricConfig) for index, metricConfig in enumerate(metric[1])))
    grid_SGD = GridSearchCV(pipeline_SGD, param_grid=param_SGD, refit=True, n_jobs=nbCores, scoring='accuracy',
                            cv=nbFolds)
    SGD_detector = grid_SGD.fit(X_train, y_train)
    desc_params = [SGD_detector.best_params_["classifier__loss"], SGD_detector.best_params_["classifier__penalty"],
                   SGD_detector.best_params_["classifier__alpha"]]
    return desc_params

def getConfig(config):
    return "\n\t\t- SGDClassifier with loss : "+config[0]+", penalty : "+config[1]