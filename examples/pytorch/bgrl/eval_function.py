import numpy as np
import torch
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder


def fit_logistic_regression(X, y, data_random_seed=1, repeat=1):
    # transform targets to one-hot vector
    one_hot_encoder = OneHotEncoder(categories="auto", sparse=False)

    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(np.bool_)

    # normalize x
    X = normalize(X, norm="l2")

    # set random state, this will ensure the dataset will be split exactly the same throughout training
    rng = np.random.RandomState(data_random_seed)

    accuracies = []
    for _ in range(repeat):
        # different random split after each repeat
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.8, random_state=rng
        )

        # grid search with one-vs-rest classifiers
        logreg = LogisticRegression(solver="liblinear")
        c = 2.0 ** np.arange(-10, 11)
        cv = ShuffleSplit(n_splits=5, test_size=0.5)
        clf = GridSearchCV(
            estimator=OneVsRestClassifier(logreg),
            param_grid=dict(estimator__C=c),
            n_jobs=5,
            cv=cv,
            verbose=0,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(
            np.bool_
        )

        test_acc = metrics.accuracy_score(y_test, y_pred)
        accuracies.append(test_acc)
    return accuracies


def fit_logistic_regression_preset_splits(
    X, y, train_mask, val_mask, test_mask
):
    # transform targets to one-hot vector
    one_hot_encoder = OneHotEncoder(categories="auto", sparse=False)
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(np.bool_)

    # normalize x
    X = normalize(X, norm="l2")

    accuracies = []
    for split_id in range(train_mask.shape[1]):
        # get train/val/test masks
        tmp_train_mask, tmp_val_mask = (
            train_mask[:, split_id],
            val_mask[:, split_id],
        )

        # make custom cv
        X_train, y_train = X[tmp_train_mask], y[tmp_train_mask]
        X_val, y_val = X[tmp_val_mask], y[tmp_val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # grid search with one-vs-rest classifiers
        best_test_acc, best_acc = 0, 0
        for c in 2.0 ** np.arange(-10, 11):
            clf = OneVsRestClassifier(
                LogisticRegression(solver="liblinear", C=c)
            )
            clf.fit(X_train, y_train)

            y_pred = clf.predict_proba(X_val)
            y_pred = np.argmax(y_pred, axis=1)
            y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(
                np.bool_
            )
            val_acc = metrics.accuracy_score(y_val, y_pred)
            if val_acc > best_acc:
                best_acc = val_acc
                y_pred = clf.predict_proba(X_test)
                y_pred = np.argmax(y_pred, axis=1)
                y_pred = one_hot_encoder.transform(
                    y_pred.reshape(-1, 1)
                ).astype(np.bool_)
                best_test_acc = metrics.accuracy_score(y_test, y_pred)

        accuracies.append(best_test_acc)
    return accuracies


def fit_ppi_linear(
    num_classes, train_data, val_data, test_data, device, repeat=1
):
    r"""
    Trains a linear layer on top of the representations. This function is specific to the PPI dataset,
    which has multiple labels.
    """

    def train(classifier, train_data, optimizer):
        classifier.train()

        x, label = train_data
        x, label = x.to(device), label.to(device)
        for step in range(100):
            # forward
            optimizer.zero_grad()
            pred_logits = classifier(x)

            # loss and backprop
            loss = criterion(pred_logits, label)
            loss.backward()
            optimizer.step()

    def test(classifier, data):
        classifier.eval()
        x, label = data
        label = label.cpu().numpy().squeeze()

        # feed to network and classifier
        with torch.no_grad():
            pred_logits = classifier(x.to(device))
            pred_class = (pred_logits > 0).float().cpu().numpy()

        return (
            metrics.f1_score(label, pred_class, average="micro")
            if pred_class.sum() > 0
            else 0
        )

    num_feats = train_data[0].size(1)
    criterion = torch.nn.BCEWithLogitsLoss()

    # normalization
    mean, std = train_data[0].mean(0, keepdim=True), train_data[0].std(
        0, unbiased=False, keepdim=True
    )
    train_data[0] = (train_data[0] - mean) / std
    val_data[0] = (val_data[0] - mean) / std
    test_data[0] = (test_data[0] - mean) / std

    best_val_f1 = []
    test_f1 = []
    for _ in range(repeat):
        tmp_best_val_f1 = 0
        tmp_test_f1 = 0
        for weight_decay in 2.0 ** np.arange(-10, 11, 2):
            classifier = torch.nn.Linear(num_feats, num_classes).to(device)
            optimizer = torch.optim.AdamW(
                params=classifier.parameters(),
                lr=0.01,
                weight_decay=weight_decay,
            )

            train(classifier, train_data, optimizer)
            val_f1 = test(classifier, val_data)
            if val_f1 > tmp_best_val_f1:
                tmp_best_val_f1 = val_f1
                tmp_test_f1 = test(classifier, test_data)
        best_val_f1.append(tmp_best_val_f1)
        test_f1.append(tmp_test_f1)

    return [best_val_f1], [test_f1]
