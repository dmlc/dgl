from statistics import mean

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegressionClassifier(nn.Module):
    """Define a logistic regression classifier to evaluate the quality of embedding results"""

    def __init__(self, nfeat, nclass):
        super(LogisticRegressionClassifier, self).__init__()
        self.lrc = nn.Linear(nfeat, nclass)

    def forward(self, x):
        preds = self.lrc(x)
        return preds


def _evaluate(model, features, labels, test_mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[test_mask]
        labels = labels[test_mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def _train_test_with_lrc(model, features, labels, train_mask, test_mask):
    """Under the pre-defined balanced train/test label setting, train a lrc to evaluate the embedding results."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.2, weight_decay=5e-06)
    for _ in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(features)
        loss_train = F.cross_entropy(output[train_mask], labels[train_mask])
        loss_train.backward()
        optimizer.step()
    return _evaluate(
        model=model, features=features, labels=labels, test_mask=test_mask
    )


def evaluate_embeds(
    features, labels, train_mask, test_mask, n_classes, cuda, test_times=10
):
    print(
        "Training a logistic regression classifier with the pre-defined train/test split setting ..."
    )
    res_list = []
    for _ in range(test_times):
        model = LogisticRegressionClassifier(
            nfeat=features.shape[1], nclass=n_classes
        )
        if cuda:
            model.cuda()
        res = _train_test_with_lrc(
            model=model,
            features=features,
            labels=labels,
            train_mask=train_mask,
            test_mask=test_mask,
        )
        res_list.append(res)
    return mean(res_list)
