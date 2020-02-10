from sklearn.metrics import make_scorer


def score(y_true, y_pred, multiclass=False, type='f1_score', **kwargs):
    """Arguments:
    y_true: real labels
    y_pred: predicted labels

    Keyword Arguments:
    "0": weights to compute accuracy

    Returns:
    Weighted accuracy score for y_true, y_pred"""
    _type_names = ['accuracy_score', 'f1_score', 'fbeta_score', 'hamming_loss',
                  'jaccard_similarity_score', 'precision_score', 'recall_score',
                  'roc_auc_score', 'zero_one_loss', 'zero_one_loss', 'framework']
    if type not in _type_names:
        raise NameError('type  must be in :' + _type_names)
    version = -1
    try:
       kwargs0  = kwargs["0"]
    except Exception:
       kwargs0  = None
    if type.startswith('matthews_corrcoef'):
        from sklearn.metrics import matthews_corrcoef
        score = matthews_corrcoef(y_true, y_pred)
    elif type.startswith('accuracy_score'):
        version = 0
        from sklearn.metrics import accuracy_score
        score = accuracy_score (y_true, y_pred, sample_weight=kwargs0)
    elif type.startswith('zero_one_loss'):
         from sklearn.metrics import zero_one_loss
         score = zero_one_loss(y_true, y_pred, sample_weight=kwargs0)
    elif type.startswith('hamming_loss'):
        from sklearn.metrics import hamming_loss
        classes = kwargs0
        score = hamming_loss(y_true, y_pred)
    elif type.startswith('f1_score'):
        version = 1
        from sklearn.metrics import f1_score

        try:
            labels = kwargs["1"]
        except:
            labels = None
        try:
            pos_label = kwargs["2"]
        except:
            pos_label = 1
        try:
            average = kwargs["3"]
        except Exception:
            if multiclass:
                average = "micro"
            else:
                average = "binary"
        score = f1_score(y_true, y_pred, sample_weight=kwargs0, labels=labels,
                   pos_label=pos_label, average=average)
    elif type.startswith('fbeta_score'):
        from sklearn.metrics import fbeta_score
        try:
            beta = kwargs["1"]
        except Exception:
            beta = 1.0
        try:
            labels = kwargs["2"]
        except Exception:
            labels = None
        try:
            pos_label = kwargs["3"]
        except Exception:
            pos_label = 1
        try:
            average = kwargs["4"]
        except Exception:
            if multiclass:
                average = "micro"
            else:
                average = "binary"
        score = fbeta_score(y_true, y_pred, beta, sample_weight=kwargs0,
                       labels=labels, pos_label=pos_label,
                       average=average)
    elif type.startswith('jaccard_similarity_score'):
       from sklearn.metrics import jaccard_similarity_score
       score = jaccard_similarity_score(y_true, y_pred, sample_weight=kwargs0)
    elif type.startswith('log_loss'):
        from sklearn.metrics import log_loss
        try:
            eps = kwargs["1"]
        except Exception:
            eps = 1e-15
        score = log_loss(y_true, y_pred, sample_weight=kwargs0, eps=eps)
    elif type.startswith('precision_score'):
         from sklearn.metrics import precision_score
         try:
             labels = kwargs["1"]
         except Exception:
             labels = None
         try:
             pos_label = kwargs["2"]
         except Exception:
             pos_label = 1
         try:
             average = kwargs["3"]
         except Exception:
             if multiclass:
                 average = "micro"
             else:
                 average = "binary"
         score = precision_score(y_true, y_pred,
                                sample_weight=kwargs0, labels=labels,
                                pos_label=pos_label, average=average)
    elif type.startswith('recall_score'):
         from sklearn.metrics import recall_score
         try:
             labels = kwargs["1"]
         except Exception:
             labels = None
         try:
             pos_label = kwargs["2"]
         except Exception:
             pos_label = 1
         try:
             average = kwargs["3"]
         except Exception:
             if multiclass:
                 average = "micro"
             else:
                 average = "binary"
         score = recall_score(y_true, y_pred, sample_weight=kwargs0,
                              labels=labels,
                              pos_label=pos_label, average=average)
    elif type.startswith('roc_auc_score'):
         from sklearn.metrics import roc_auc_score
         from sklearn.preprocessing import MultiLabelBinarizer
         try:
             average = kwargs["1"]
         except Exception:
             if multiclass:
                 average = "micro"
             else:
                 average = None
         if multiclass:
             mlb = MultiLabelBinarizer()
             y_true = mlb.fit_transform([(label) for label in y_true])
             y_pred = mlb.fit_transform([(label) for label in y_pred])
         score = roc_auc_score(y_true, y_pred,
                               sample_weight=kwargs0, average=average)
    else:
        score = 0.0
        return score




def get_scorer(type='f1_score', **kwargs):
    """Keyword Arguments:
    "0": weights to compute accuracy

    Returns:
    A weighted sklearn scorer for accuracy"""
    _type_names = ['accuracy_score', 'f1_score', 'fbeta_score', 'hamming_loss',
                  'jaccard_similarity_score', 'precision_score', 'recall_score',
                  'roc_auc_score', 'zero_one_loss', 'zero_one_loss', 'framework']
    if type not in _type_names:
        raise NameError('type  must be in :' + _type_names)
    try:
        sample_weight = kwargs["0"]
    except Exception:
        sample_weight = None
    if  type.startswith('accuracy_score'):
        version = 0
        from sklearn.metrics import accuracy_score as metric
        return make_scorer(metric, greater_is_better=True,
                           sample_weight=sample_weight)
    elif type.startswith('f1_score'):
        try:
            labels = kwargs["1"]
        except Exception:
            labels = None
        try:
            pos_label = kwargs["2"]
        except Exception:
            pos_label = 1
        try:
            average = kwargs["3"]
        except:
            average = "binary"
        from sklearn.metrics import f1_score as metric
        return make_scorer(metric, greater_is_better=True,
                       sample_weight=sample_weight, labels=labels,
                       pos_label=pos_label, average=average)
    elif type.startswith('fbeta_score'):
        try:
            beta = kwargs["1"]
        except Exception:
            beta = 1.0
        try:
            labels = kwargs["2"]
        except Exception:
            labels = None
        try:
            pos_label = kwargs["3"]
        except Exception:
            pos_label = 1
        try:
            average = kwargs["4"]
        except Exception:
            average = "binary"
        from sklearn.metrics import fbeta_score as metric
        return make_scorer(metric, greater_is_better=True, beta=beta,
                       sample_weight=sample_weight, labels=labels,
                       pos_label=pos_label, average=average)
    elif type.startswith('hamming_loss'):
        try:
            classes = kwargs["0"]
        except Exception:
            classes = None
        from sklearn.metrics import hamming_loss as metric
        return make_scorer(metric, greater_is_better=False, classes=classes)
    elif type.startswith('jaccard_similarity_score'):
        from sklearn.metrics import jaccard_similarity_score as metric
        return make_scorer(metric, greater_is_better=True,
                           sample_weight=sample_weight)

    elif type.startswith('log_loss'):
        from sklearn.metrics import log_loss as metric

        try:
            eps = kwargs["1"]
        except Exception:
            eps = 1e-15
        return make_scorer(metric, greater_is_better=False,
                   sample_weight=sample_weight, eps=eps)
    elif type.startswith('matthews_corrcoef'):
        from sklearn.metrics import matthews_corrcoef as metric
        return make_scorer(metric, greater_is_better=True)

    elif type.startswith('precision_score'):
        from sklearn.metrics import precision_score as metric
        try:
            labels = kwargs["1"]
        except Exception:
            labels = None
        try:
            pos_label = kwargs["2"]
        except Exception:
            pos_label = 1
        try:
            average = kwargs["3"]
        except Exception:
            average = "binary"
        return make_scorer(metric, greater_is_better=True,
                       sample_weight=sample_weight, labels=labels,
                       pos_label=pos_label,
                       average=average)
    elif type.startswith('recall_score'):
        try:
            sample_weight = kwargs["0"]
        except Exception:
            sample_weight = None
        try:
            labels = kwargs["1"]
        except Exception:
            labels = None
        try:
            pos_label = kwargs["2"]
        except Exception:
            pos_label = 1
        try:
            average = kwargs["3"]
        except Exception:
            average = "binary"
        from sklearn.metrics import recall_score as metric
        return make_scorer(metric, greater_is_better=True,
                           sample_weight=sample_weight, labels=labels,
                           pos_label=pos_label,
                           average=average)
    elif type.startswith('roc_auc_score'):
        try:
            average = kwargs["1"]
        except:
            average = "micro"
        from sklearn.metrics import roc_auc_score as metric
        return make_scorer(metric, greater_is_better=True,
                           sample_weight=sample_weight, average=average)
    elif type.startswith('zero_one_loss'):
        from sklearn.metrics import zero_one_loss as metric
        return make_scorer(metric, greater_is_better=False,
                       sample_weight=sample_weight)
    else:
        scorer = None
        return scorer


def get_config(type='f1_score', **kwargs):
    _type_names = ['accuracy_score', 'f1_score', 'fbeta_score', 'hamming_loss',
                  'jaccard_similarity_score', 'precision_score', 'recall_score',
                  'roc_auc_score', 'zero_one_loss', 'zero_one_loss', 'framework']
    if type not in _type_names:
        raise NameError('type  must be in :' + _type_names)
    try:
        sample_weight = kwargs["0"]
    except Exception:
        sample_weight = None
    if  type.startswith('accuracy_score'):
        config_string = "Accuracy score using " + str(
            sample_weight) + " as sample_weights (higher is better)"
    elif type.startswith('f1_score'):
        try:
            labels = kwargs["1"]
        except Exception:
            labels = None
        try:
            pos_label = kwargs["2"]
        except Exception:
            pos_label = 1
        try:
            average = kwargs["3"]
        except Exception:
            average = "binary"
        config_string = "F1 score using " + str(
            sample_weight) + " as sample_weights, " + str(
            labels) + " as labels, " + str(
            pos_label) \
                        + " as pos_label, " + average + " as average (higher is better)"

    elif type.startswith('fbeta_score'):
        try:
            beta = kwargs["1"]
        except Exception:
            beta = 1.0
        try:
            labels = kwargs["1"]
        except Exception:
            labels = None
        try:
            pos_label = kwargs["2"]
        except Exception:
            pos_label = 1
        try:
            average = kwargs["3"]
        except Exception:
            average = "binary"
        config_string = "F-beta score using " + str(
            sample_weight) + " as sample_weights, " + str(
            labels) + " as labels, " + str(pos_label) \
                        + " as pos_label, " + average + " as average, " + str(
            beta) + " as beta (higher is better)"
    elif type.startswith('hamming_loss'):
        try:
            classes = kwargs["0"]
        except Exception:
            classes = None
        config_string = "Hamming loss using " + str(
            classes) + " as classes (lower is better)"
    elif type.startswith('jaccard_similarity_score'):
        config_string = "Jaccard_similarity score using " + str(
            sample_weight) + " as sample_weights (higher is better)"
    elif type.startswith('log_loss'):
        try:
            eps = kwargs["1"]
        except Exception:
            eps = 1e-15
        config_string = "Log loss using " + str(
            sample_weight) + " as sample_weights, " + str(
            eps) + " as eps (lower is better)"
    elif type.startswith('matthews_corrcoef'):
        config_string = "Matthews correlation coefficient (higher is better)"
    elif type.startswith('precision_score'):
        try:
            labels = kwargs["1"]
        except Exception:
            labels = None
        try:
            pos_label = kwargs["2"]
        except Exception:
            pos_label = 1
        try:
            average = kwargs["3"]
        except:
            average = "binary"
        config_string = "Precision score using " + str(
            sample_weight) + " as sample_weights, " + str(
            labels) + " as labels, " + str(pos_label) \
                        + " as pos_label, " + average + " as average (higher is better)"
    elif type.startswith('recall_score'):
        try:
            labels = kwargs["1"]
        except Exception:
            labels = None
        try:
            pos_label = kwargs["2"]
        except Exception:
            pos_label = 1
        try:
            average = kwargs["3"]
        except Exception:
            average = "binary"
        configString = "Recall score using " + str(
            sample_weight) + " as sample_weights, " + str(
            labels) + " as labels, " + str(pos_label) \
                       + " as pos_label, " + average + "as average (higher is " \
                                                       "better) "
    elif type.startswith('roc_auc_score'):
        configString = "ROC_AUC score using " + str(
            sample_weight) + " as sample_weights, " + average + " as average (higher is better)"
    elif type.startswith('zero_one_loss'):
        configString = "Zero_one loss using " + str(
            sample_weight) + " as sample_weights (lower is better)"
    else:
        config_tring = "This is a framework"
    return configString
