import numpy as np

from pyscm.scm import SetCoveringMachineClassifier as scm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.externals.six import iteritems, iterkeys, itervalues

from sklearn.metrics import accuracy_score
import itertools

from ..LateFusion import LateFusionClassifier, getClassifiers, getConfig
from ..... import MonoviewClassifiers
from .....utils.Dataset import getV


class DecisionStumpSCMNew(BaseEstimator, ClassifierMixin):
    """docstring for SCM
    A hands on class of SCM using decision stump, built with sklearn format in order to use sklearn function on SCM like
    CV, gridsearch, and so on ..."""

    def __init__(self, model_type='conjunction', p=0.1, max_rules=10, random_state=42):
        super(DecisionStumpSCMNew, self).__init__()
        self.model_type = model_type
        self.p = p
        self.max_rules = max_rules
        self.random_state = random_state

    def fit(self, X, y):
        self.clf = scm(model_type=self.model_type, max_rules=self.max_rules, p=self.p, random_state=self.random_state)
        self.clf.fit(X=X, y=y)

    def predict(self, X):
        return self.clf.predict(X)

    def set_params(self, **params):
        for key, value in iteritems(params):
            if key == 'p':
                self.p = value
            if key == 'model_type':
                self.model_type = value
            if key == 'max_rules':
                self.max_rules = value

    def get_stats(self):
        return {"Binary_attributes": self.clf.model_.rules}


def genParamsSets(classificationKWARGS, randomState, nIter=1):
    nbView = classificationKWARGS["nbView"]
    paramsSets = []
    for _ in range(nIter):
        max_attributes = randomState.randint(1, 20)
        p = randomState.random_sample()
        model = randomState.choice(["conjunction", "disjunction"])
        order = randomState.randint(1, 10)
        paramsSets.append([p, max_attributes, model, order])
    return paramsSets


def getArgs(benchmark, args, views, viewsIndices, directory, resultsMonoview, classificationIndices):
    if args.FU_L_cl_names != ['']:
        pass
    else:
        monoviewClassifierModulesNames = benchmark["Monoview"]
        args.FU_L_cl_names = getClassifiers(args.FU_L_select_monoview, monoviewClassifierModulesNames, directory,
                                            viewsIndices, resultsMonoview, classificationIndices)
    monoviewClassifierModules = [getattr(MonoviewClassifiers, classifierName)
                                 for classifierName in args.FU_L_cl_names]
    if args.FU_L_cl_names == [""] and args.CL_type == ["Multiview"]:
        raise AttributeError("You must perform Monoview classification or specify "
                             "which monoview classifier to use Late Fusion")
    if args.FU_L_cl_config != ['']:
        classifiersConfigs = [
            monoviewClassifierModule.getKWARGS([arg.split(":") for arg in classifierConfig.split(",")])
            for monoviewClassifierModule, classifierConfig
            in zip(monoviewClassifierModules, args.FU_L_cl_config)]
    else:
        classifiersConfigs = getConfig(args.FU_L_cl_names, resultsMonoview)
    arguments = {"CL_type": "Fusion",
                 "views": views,
                 "NB_VIEW": len(views),
                 "viewsIndices": viewsIndices,
                 "NB_CLASS": len(args.CL_classes),
                 "LABELS_NAMES": args.CL_classes,
                 "FusionKWARGS": {"fusionType": "LateFusion",
                                  "fusionMethod": "SCMForLinear",
                                  "classifiersNames": args.FU_L_cl_names,
                                  "classifiersConfigs": classifiersConfigs,
                                  'fusionMethodConfig': args.FU_L_method_config,
                                  'monoviewSelection': args.FU_L_select_monoview,
                                  "nbView": (len(viewsIndices))}}
    return [arguments]


class SCMForLinear(LateFusionClassifier):
    def __init__(self, randomState, NB_CORES=1, **kwargs):
        LateFusionClassifier.__init__(self, randomState, kwargs['classifiersNames'], kwargs['classifiersConfigs'],
                                      kwargs["monoviewSelection"],
                                      NB_CORES=NB_CORES)
        self.SCMClassifier = None
        if kwargs['fusionMethodConfig'][0] is None or kwargs['fusionMethodConfig'] == ['']:
            self.p = 1
            self.maxAttributes = 5
            self.order = 1
            self.modelType = "conjunction"
        else:
            self.p = int(kwargs['fusionMethodConfig'][0])
            self.maxAttributes = int(kwargs['fusionMethodConfig'][1])
            self.order = int(kwargs['fusionMethodConfig'][2])
            self.modelType = kwargs['fusionMethodConfig'][3]

    def setParams(self, paramsSet):
        self.p = paramsSet[0]
        self.maxAttributes = paramsSet[1]
        self.order = paramsSet[3]
        self.modelType = paramsSet[2]

    def fit_hdf5(self, DATASET, trainIndices=None, viewsIndices=None):
        if viewsIndices is None:
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
        if trainIndices is None:
            trainIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        for index, viewIndex in enumerate(viewsIndices):
            monoviewClassifier = getattr(MonoviewClassifiers, self.monoviewClassifiersNames[index])
            self.monoviewClassifiers.append(
                monoviewClassifier.fit(getV(DATASET, viewIndex, trainIndices),
                                       DATASET.get("Labels").value[trainIndices], self.randomState,
                                       NB_CORES=self.nbCores,
                                       **self.monoviewClassifiersConfigs[index]))
        self.SCMForLinearFusionFit(DATASET, usedIndices=trainIndices, viewsIndices=viewsIndices)

    def predict_hdf5(self, DATASET, usedIndices=None, viewsIndices=None):
        if viewsIndices is None:
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
        nbView = len(viewsIndices)
        if usedIndices is None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        monoviewDecisions = np.zeros((len(usedIndices), nbView), dtype=int)
        accus = []
        for index, viewIndex in enumerate(viewsIndices):
            monoviewDecision = self.monoviewClassifiers[index].predict(
                getV(DATASET, viewIndex, usedIndices))
            accus.append(accuracy_score(DATASET.get("Labels").value[usedIndices], monoviewDecision))
            monoviewDecisions[:, index] = monoviewDecision
        features = self.generateInteractions(monoviewDecisions)
        predictedLabels = self.SCMClassifier.predict(features)
        return predictedLabels

    def SCMForLinearFusionFit(self, DATASET, usedIndices=None, viewsIndices=None):
        if type(viewsIndices) == type(None):
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])

        nbView = len(viewsIndices)
        self.SCMClassifier = DecisionStumpSCMNew(p=self.p, max_rules=self.maxAttributes, model_type=self.modelType,
                                                 random_state=self.randomState)
        monoViewDecisions = np.zeros((len(usedIndices), nbView), dtype=int)
        for index, viewIndex in enumerate(viewsIndices):
            monoViewDecisions[:, index] = self.monoviewClassifiers[index].predict(
                getV(DATASET, viewIndex, usedIndices))
        features = self.generateInteractions(monoViewDecisions)
        features = np.array([np.array([feat for feat in feature]) for feature in features])
        self.SCMClassifier.fit(features, DATASET.get("Labels").value[usedIndices].astype(int))

    def generateInteractions(self, monoViewDecisions):
        if type(self.order) == type(None):
            self.order = monoViewDecisions.shape[1]
        if self.order == 1:
            return monoViewDecisions
        else:
            genratedIntercations = [monoViewDecisions[:, i] for i in range(monoViewDecisions.shape[1])]
            for orderIndex in range(self.order - 1):
                combins = itertools.combinations(range(monoViewDecisions.shape[1]), orderIndex + 2)
                for combin in combins:
                    generatedDecision = monoViewDecisions[:, combin[0]]
                    for index in range(len(combin) - 1):
                        if self.modelType == "disjunction":
                            generatedDecision = np.logical_and(generatedDecision,
                                                               monoViewDecisions[:, combin[index + 1]])
                        else:
                            generatedDecision = np.logical_or(generatedDecision,
                                                              monoViewDecisions[:, combin[index + 1]])
                    genratedIntercations.append(generatedDecision)
            return np.transpose(np.array(genratedIntercations))

    def getConfig(self, fusionMethodConfig, monoviewClassifiersNames, monoviewClassifiersConfigs):
        configString = "with SCM for linear with max_attributes : " + str(self.maxAttributes) + ", p : " + str(self.p) + \
                       " model_type : " + str(self.modelType) + " order : " + str(self.order)+ " has chosen " + \
                       str(0.1) + " rule(s) \n\t-With monoview classifiers : "
        for monoviewClassifierConfig, monoviewClassifierName in zip(monoviewClassifiersConfigs,
                                                                    monoviewClassifiersNames):
            monoviewClassifierModule = getattr(MonoviewClassifiers, monoviewClassifierName)
            configString += monoviewClassifierModule.getConfig(monoviewClassifierConfig)
        return configString