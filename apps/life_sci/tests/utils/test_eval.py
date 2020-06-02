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
    meter.update(pred, label)
    true_scores = [0.7500000774286983, 0.7500000516191412]
    assert meter.pearson_r2() == true_scores
    assert meter.pearson_r2('mean') == np.mean(true_scores)
    assert meter.pearson_r2('sum') == np.sum(true_scores)
    assert meter.compute_metric('r2') == true_scores
    assert meter.compute_metric('r2', 'mean') == np.mean(true_scores)
    assert meter.compute_metric('r2', 'sum') == np.sum(true_scores)

    meter = Meter(label_mean, label_std)
    meter.update(pred, label, mask)
    true_scores = [1.0, 1.0]
    assert meter.pearson_r2() == true_scores
    assert meter.pearson_r2('mean') == np.mean(true_scores)
    assert meter.pearson_r2('sum') == np.sum(true_scores)
    assert meter.compute_metric('r2') == true_scores
    assert meter.compute_metric('r2', 'mean') == np.mean(true_scores)
    assert meter.compute_metric('r2', 'sum') == np.sum(true_scores)

    # mae
    meter = Meter()
    meter.update(pred, label)
    true_scores = [0.1666666716337204, 0.1666666716337204]
    assert meter.mae() == true_scores
    assert meter.mae('mean') == np.mean(true_scores)
    assert meter.mae('sum') == np.sum(true_scores)
    assert meter.compute_metric('mae') == true_scores
    assert meter.compute_metric('mae', 'mean') == np.mean(true_scores)
    assert meter.compute_metric('mae', 'sum') == np.sum(true_scores)

    meter = Meter()
    meter.update(pred, label, mask)
    true_scores = [0.25, 0.0]
    assert meter.mae() == true_scores
    assert meter.mae('mean') == np.mean(true_scores)
    assert meter.mae('sum') == np.sum(true_scores)
    assert meter.compute_metric('mae') == true_scores
    assert meter.compute_metric('mae', 'mean') == np.mean(true_scores)
    assert meter.compute_metric('mae', 'sum') == np.sum(true_scores)

    # rmsef
    meter = Meter(label_mean, label_std)
    meter.update(pred, label)
    true_scores = [0.41068359261794546, 0.4106836107598449]
    assert torch.allclose(torch.tensor(meter.rmse()), torch.tensor(true_scores))
    assert torch.allclose(torch.tensor(meter.compute_metric('rmse')), torch.tensor(true_scores))

    meter = Meter(label_mean, label_std)
    meter.update(pred, label, mask)
    true_scores = [0.44433766459035057, 0.5019903799993205]
    assert torch.allclose(torch.tensor(meter.rmse()), torch.tensor(true_scores))
    assert torch.allclose(torch.tensor(meter.compute_metric('rmse')), torch.tensor(true_scores))

    # roc auc score
    meter = Meter()
    meter.update(pred, label)
    true_scores = [1.0, 1.0]
    assert meter.roc_auc_score() == true_scores
    assert meter.roc_auc_score('mean') == np.mean(true_scores)
    assert meter.roc_auc_score('sum') == np.sum(true_scores)
    assert meter.compute_metric('roc_auc_score') == true_scores
    assert meter.compute_metric('roc_auc_score', 'mean') == np.mean(true_scores)
    assert meter.compute_metric('roc_auc_score', 'sum') == np.sum(true_scores)

    meter = Meter()
    meter.update(pred, label, mask)
    true_scores = [1.0, 1.0]
    assert meter.roc_auc_score() == true_scores
    assert meter.roc_auc_score('mean') == np.mean(true_scores)
    assert meter.roc_auc_score('sum') == np.sum(true_scores)
    assert meter.compute_metric('roc_auc_score') == true_scores
    assert meter.compute_metric('roc_auc_score', 'mean') == np.mean(true_scores)
    assert meter.compute_metric('roc_auc_score', 'sum') == np.sum(true_scores)

def test_cases_with_undefined_scores():
    label = torch.tensor([[0., 1.],
                          [0., 1.],
                          [1., 1.]])
    pred = torch.tensor([[0.5, 0.5],
                         [0., 1.],
                         [1., 0.]])
    meter = Meter()
    meter.update(pred, label)
    true_scores = [1.0]
    assert meter.roc_auc_score() == true_scores
    assert meter.roc_auc_score('mean') == np.mean(true_scores)
    assert meter.roc_auc_score('sum') == np.sum(true_scores)

if __name__ == '__main__':
    test_Meter()
    test_cases_with_undefined_scores()
