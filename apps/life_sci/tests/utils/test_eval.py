import numpy as np
import torch

from dgllife.utils.eval import *

def test_Meter():
    label = torch.tensor([[0., 1.],
                          [0., 1.],
                          [1., 0.]])
    pred = torch.tensor([[0.5, 0.5],
                         [0., 1.],
                         [1., 0.]])
    mask = torch.tensor([[1., 0.],
                         [0., 1.],
                         [1., 1.]])
    label_mean, label_std = label.mean(dim=0), label.std(dim=0)

    # pearson r2
    meter = Meter(label_mean, label_std)
    meter.update(label, pred)
    true_scores = [0.7499999999999999, 0.7499999999999999]
    assert meter.pearson_r2() == true_scores
    assert meter.pearson_r2('mean') == np.mean(true_scores)
    assert meter.pearson_r2('sum') == np.sum(true_scores)
    assert meter.compute_metric('r2') == true_scores
    assert meter.compute_metric('r2', 'mean') == np.mean(true_scores)
    assert meter.compute_metric('r2', 'sum') == np.sum(true_scores)

    meter = Meter(label_mean, label_std)
    meter.update(label, pred, mask)
    true_scores = [1.0, 1.0]
    assert meter.pearson_r2() == true_scores
    assert meter.pearson_r2('mean') == np.mean(true_scores)
    assert meter.pearson_r2('sum') == np.sum(true_scores)
    assert meter.compute_metric('r2') == true_scores
    assert meter.compute_metric('r2', 'mean') == np.mean(true_scores)
    assert meter.compute_metric('r2', 'sum') == np.sum(true_scores)

    # mae
    meter = Meter()
    meter.update(label, pred)
    true_scores = [0.1666666716337204, 0.1666666716337204]
    assert meter.mae() == true_scores
    assert meter.mae('mean') == np.mean(true_scores)
    assert meter.mae('sum') == np.sum(true_scores)
    assert meter.compute_metric('mae') == true_scores
    assert meter.compute_metric('mae', 'mean') == np.mean(true_scores)
    assert meter.compute_metric('mae', 'sum') == np.sum(true_scores)

    meter = Meter()
    meter.update(label, pred, mask)
    true_scores = [0.25, 0.0]
    assert meter.mae() == true_scores
    assert meter.mae('mean') == np.mean(true_scores)
    assert meter.mae('sum') == np.sum(true_scores)
    assert meter.compute_metric('mae') == true_scores
    assert meter.compute_metric('mae', 'mean') == np.mean(true_scores)
    assert meter.compute_metric('mae', 'sum') == np.sum(true_scores)

    # rmse
    meter = Meter(label_mean, label_std)
    meter.update(label, pred)
    true_scores = [0.22125875529784111, 0.5937311018897714]
    assert meter.rmse() == true_scores
    assert meter.rmse('mean') == np.mean(true_scores)
    assert meter.rmse('sum') == np.sum(true_scores)
    assert meter.compute_metric('rmse') == true_scores
    assert meter.compute_metric('rmse', 'mean') == np.mean(true_scores)
    assert meter.compute_metric('rmse', 'sum') == np.sum(true_scores)

    meter = Meter(label_mean, label_std)
    meter.update(label, pred, mask)
    true_scores = [0.1337071188699867, 0.5019903799993205]
    assert meter.rmse() == true_scores
    assert meter.rmse('mean') == np.mean(true_scores)
    assert meter.rmse('sum') == np.sum(true_scores)
    assert meter.compute_metric('rmse') == true_scores
    assert meter.compute_metric('rmse', 'mean') == np.mean(true_scores)
    assert meter.compute_metric('rmse', 'sum') == np.sum(true_scores)

    # roc auc score
    meter = Meter()
    meter.update(label, pred)
    true_scores = [1.0, 0.75]
    assert meter.roc_auc_score() == true_scores
    assert meter.roc_auc_score('mean') == np.mean(true_scores)
    assert meter.roc_auc_score('sum') == np.sum(true_scores)
    assert meter.compute_metric('roc_auc_score') == true_scores
    assert meter.compute_metric('roc_auc_score', 'mean') == np.mean(true_scores)
    assert meter.compute_metric('roc_auc_score', 'sum') == np.sum(true_scores)

    meter = Meter()
    meter.update(label, pred, mask)
    true_scores = [1.0, 1.0]
    assert meter.roc_auc_score() == true_scores
    assert meter.roc_auc_score('mean') == np.mean(true_scores)
    assert meter.roc_auc_score('sum') == np.sum(true_scores)
    assert meter.compute_metric('roc_auc_score') == true_scores
    assert meter.compute_metric('roc_auc_score', 'mean') == np.mean(true_scores)
    assert meter.compute_metric('roc_auc_score', 'sum') == np.sum(true_scores)

if __name__ == '__main__':
    test_Meter()
