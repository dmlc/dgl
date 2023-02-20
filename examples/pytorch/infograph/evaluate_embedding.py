""" Evaluate unsupervised embedding using a variety of basic classifiers. """
""" Credit: https://github.com/fanyun-sun/InfoGraph """

import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


def logistic_classify(x, y, device="cpu"):
    nb_classes = np.unique(y).shape[0]
    xent = nn.CrossEntropyLoss()
    hid_units = x.shape[1]

    accs = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    for train_index, test_index in kf.split(x, y):
        train_embs, test_embs = x[train_index], x[test_index]
        train_lbls, test_lbls = y[train_index], y[test_index]

        train_embs, train_lbls = torch.from_numpy(train_embs).to(
            device
        ), torch.from_numpy(train_lbls).to(device)
        test_embs, test_lbls = torch.from_numpy(test_embs).to(
            device
        ), torch.from_numpy(test_lbls).to(device)

        log = LogReg(hid_units, nb_classes)
        log = log.to(device)

        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        for it in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc.item())
    return np.mean(accs)


def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if search:
            params = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(
                SVC(), params, cv=5, scoring="accuracy", verbose=0
            )
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies)


def evaluate_embedding(embeddings, labels, search=True, device="cpu"):
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)

    logreg_accuracy = logistic_classify(x, y, device)
    print("LogReg", logreg_accuracy)
    svc_accuracy = svc_classify(x, y, search)
    print("svc", svc_accuracy)

    return logreg_accuracy, svc_accuracy
