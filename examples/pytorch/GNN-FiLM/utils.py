import numpy as np
from sklearn.metrics import f1_score


# function to compute f1 score
def evaluate_f1_score(pred, label):
    pred = np.round(pred, 0).astype(np.int16)
    pred = pred.flatten()
    label = label.flatten()
    return f1_score(y_pred=pred, y_true=label)
