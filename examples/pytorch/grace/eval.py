"""
Code adapted from https://github.com/CRIPAC-DIG/GRACE
Linear evaluation on learned node embeddings
"""

import functools

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder


def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                }
            print_statistics(statistics, f.__name__)
            return statistics

        return wrapper

    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool_)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f"(E) | {function_name}:", end=" ")
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]["mean"]
        std = statistics[key]["std"]
        print(f"{key}={mean:.4f}+-{std:.4f}", end="")
        if i != len(statistics.keys()) - 1:
            print(",", end=" ")
        else:
            print()


@repeat(3)
def label_classification(
    embeddings, y, train_mask, test_mask, split="random", ratio=0.1
):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories="auto").fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool_)

    X = normalize(X, norm="l2")

    if split == "random":
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=1 - ratio
        )
    elif split == "public":
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = Y[train_mask]
        y_test = Y[test_mask]

    logreg = LogisticRegression(solver="liblinear")
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(
        estimator=OneVsRestClassifier(logreg),
        param_grid=dict(estimator__C=c),
        n_jobs=8,
        cv=5,
        verbose=0,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")

    return {"F1Mi": micro, "F1Ma": macro}
