from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


def evaluate_auc(pred, label):
    res = roc_auc_score(y_score=pred, y_true=label)
    return res


def evaluate_acc(pred, label):
    res = []
    for _value in pred:
        res.append(1 if _value >= 0.5 else 0)
    return accuracy_score(y_pred=res, y_true=label)


def evaluate_f1_score(pred, label):
    res = []
    for _value in pred:
        res.append(1 if _value >= 0.5 else 0)
    return f1_score(y_pred=res, y_true=label)


def evaluate_logloss(pred, label):
    res = log_loss(y_true=label, y_pred=pred, normalize=True)
    return res
