from sklearn.svm import SVC
from sklearn.pipeline import Pipeline                   # Pipelining in classification
from sklearn.grid_search import RandomizedSearchCV
import Metrics
from scipy.stats import randint

def fit(DATASET, CLASS_LABELS, NB_CORES=1,**kwargs):
    C = int(kwargs['0'])
    degree = int(kwargs['1'])
    classifier = SVC(C=C, kernel='poly', degree=degree, probability=True, max_iter=1000)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


# def fit_gridsearch(X_train, y_train, nbFolds=4, nbCores=1, metric=["accuracy_score", None], **kwargs):
#     pipeline_SVMPoly = Pipeline([('classifier', SVC(kernel="poly"))])
#     param_SVMPoly= {"classifier__C": np.random.randint(1,2000,30), "classifier__degree": np.random.randint(1,10,5)}
#     metricModule = getattr(Metrics, metric[0])
#     scorer = metricModule.get_scorer(dict((index, metricConfig) for index, metricConfig in enumerate(metric[1])))
#     grid_SVMPoly = GridSearchCV(pipeline_SVMPoly, param_grid=param_SVMPoly, refit=True, n_jobs=nbCores, scoring='accuracy',
#                                   cv=nbFolds)
#     SVMPoly_detector = grid_SVMPoly.fit(X_train, y_train)
#     desc_params = [SVMPoly_detector.best_params_["classifier__C"], SVMPoly_detector.best_params_["classifier__degree"]]
#     return desc_params


def gridSearch(X_train, y_train, nbFolds=4, nbCores=1, metric=["accuracy_score", None], nIter=30):
    pipeline_SVMPoly = Pipeline([('classifier', SVC(kernel="poly", max_iter=1000))])
    param_SVMPoly = {"classifier__C": randint(1, 10000), "classifier__degree":randint(1, 30)}
    metricModule = getattr(Metrics, metric[0])
    if metric[1]!=None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}
    scorer = metricModule.get_scorer(**metricKWARGS)
    grid_SVMPoly = RandomizedSearchCV(pipeline_SVMPoly, n_iter=nIter, param_distributions=param_SVMPoly, refit=True, n_jobs=nbCores, scoring=scorer,
                               cv=nbFolds)
    SVMRBF_detector = grid_SVMPoly.fit(X_train, y_train)
    desc_params = [SVMRBF_detector.best_params_["classifier__C"], SVMRBF_detector.best_params_["classifier__degree"]]
    return desc_params


def getConfig(config):
    try:
        return "\n\t\t- SVM Linear with C : "+str(config[0])
    except:
        return "\n\t\t- SVM Linear with C : "+str(config["0"])