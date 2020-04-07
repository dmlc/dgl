"""Evaluation of model performance."""
# pylint: disable= no-member, arguments-differ, invalid-name
import numpy as np
import torch
import torch.nn.functional as F

from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

__all__ = ['Meter']

# pylint: disable=E1101
class Meter(object):
    """Track and summarize model performance on a dataset for (multi-label) prediction.

    When dealing with multitask learning, quite often we normalize the labels so they are
    roughly at a same scale. During the evaluation, we need to undo the normalization on
    the predicted labels. If mean and std are not None, we will undo the normalization.

    Currently we support evaluation with 4 metrics:

    * ``pearson r2``
    * ``mae``
    * ``rmse``
    * ``roc auc score``

    Parameters
    ----------
    mean : torch.float32 tensor of shape (T) or None.
        Mean of existing training labels across tasks if not ``None``. ``T`` for the
        number of tasks. Default to ``None`` and we assume no label normalization has been
        performed.
    std : torch.float32 tensor of shape (T)
        Std of existing training labels across tasks if not ``None``. Default to ``None``
        and we assume no label normalization has been performed.

    Examples
    --------
    Below gives a demo for a fake evaluation epoch.

    >>> import torch
    >>> from dgllife.utils import Meter

    >>> meter = Meter()
    >>> # Simulate 10 fake mini-batches
    >>> for batch_id in range(10):
    >>>     batch_label = torch.randn(3, 3)
    >>>     batch_pred = torch.randn(3, 3)
    >>>     meter.update(batch_pred, batch_label)

    >>> # Get MAE for all tasks
    >>> print(meter.compute_metric('mae'))
    [1.1325558423995972, 1.0543707609176636, 1.094650149345398]
    >>> # Get MAE averaged over all tasks
    >>> print(meter.compute_metric('mae', reduction='mean'))
    1.0938589175542195
    >>> # Get the sum of MAE over all tasks
    >>> print(meter.compute_metric('mae', reduction='sum'))
    3.2815767526626587
    """
    def __init__(self, mean=None, std=None):
        self.mask = []
        self.y_pred = []
        self.y_true = []

        if (mean is not None) and (std is not None):
            self.mean = mean.cpu()
            self.std = std.cpu()
        else:
            self.mean = None
            self.std = None

    def update(self, y_pred, y_true, mask=None):
        """Update for the result of an iteration

        Parameters
        ----------
        y_pred : float32 tensor
            Predicted labels with shape ``(B, T)``,
            ``B`` for number of graphs in the batch and ``T`` for the number of tasks
        y_true : float32 tensor
            Ground truth labels with shape ``(B, T)``
        mask : None or float32 tensor
            Binary mask indicating the existence of ground truth labels with
            shape ``(B, T)``. If None, we assume that all labels exist and create
            a one-tensor for placeholder.
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        if mask is None:
            self.mask.append(torch.ones(self.y_pred[-1].shape))
        else:
            self.mask.append(mask.detach().cpu())

    def _finalize(self):
        """Prepare for evaluation.

        If normalization was performed on the ground truth labels during training,
        we need to undo the normalization on the predicted labels.

        Returns
        -------
        mask : float32 tensor
            Binary mask indicating the existence of ground
            truth labels with shape (B, T), B for batch size
            and T for the number of tasks
        y_pred : float32 tensor
            Predicted labels with shape (B, T)
        y_true : float32 tensor
            Ground truth labels with shape (B, T)
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)

        if (self.mean is not None) and (self.std is not None):
            # To compensate for the imbalance between labels during training,
            # we normalize the ground truth labels with training mean and std.
            # We need to undo that for evaluation.
            y_pred = y_pred * self.std + self.mean

        return mask, y_pred, y_true

    def _reduce_scores(self, scores, reduction='none'):
        """Finalize the scores to return.

        Parameters
        ----------
        scores : list of float
            Scores for all tasks.
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        if reduction == 'none':
            return scores
        elif reduction == 'mean':
            return np.mean(scores)
        elif reduction == 'sum':
            return np.sum(scores)
        else:
            raise ValueError(
                "Expect reduction to be 'none', 'mean' or 'sum', got {}".format(reduction))

    def multilabel_score(self, score_func, reduction='none'):
        """Evaluate for multi-label prediction.

        Parameters
        ----------
        score_func : callable
            A score function that takes task-specific ground truth and predicted labels as
            input and return a float as the score. The labels are in the form of 1D tensor.
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        mask, y_pred, y_true = self._finalize()
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(score_func(task_y_true, task_y_pred))
        return self._reduce_scores(scores, reduction)

    def pearson_r2(self, reduction='none'):
        """Compute squared Pearson correlation coefficient.

        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        def score(y_true, y_pred):
            return pearsonr(y_true.numpy(), y_pred.numpy())[0] ** 2
        return self.multilabel_score(score, reduction)

    def mae(self, reduction='none'):
        """Compute mean absolute error.

        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        def score(y_true, y_pred):
            return F.l1_loss(y_true, y_pred).data.item()
        return self.multilabel_score(score, reduction)

    def rmse(self, reduction='none'):
        """Compute root mean square error.

        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        def score(y_true, y_pred):
            return np.sqrt(F.mse_loss(y_pred, y_true).cpu().item())
        return self.multilabel_score(score, reduction)

    def roc_auc_score(self, reduction='none'):
        """Compute roc-auc score for binary classification.

        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        # Todo: This function only supports binary classification and we may need
        #  to support categorical classes.
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'
        def score(y_true, y_pred):
            return roc_auc_score(y_true.long().numpy(), torch.sigmoid(y_pred).numpy())
        return self.multilabel_score(score, reduction)

    def compute_metric(self, metric_name, reduction='none'):
        """Compute metric based on metric name.

        Parameters
        ----------
        metric_name : str

            * ``'r2'``: compute squared Pearson correlation coefficient
            * ``'mae'``: compute mean absolute error
            * ``'rmse'``: compute root mean square error
            * ``'roc_auc_score'``: compute roc-auc score

        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        if metric_name == 'r2':
            return self.pearson_r2(reduction)
        elif metric_name == 'mae':
            return self.mae(reduction)
        elif metric_name == 'rmse':
            return self.rmse(reduction)
        elif metric_name == 'roc_auc_score':
            return self.roc_auc_score(reduction)
        else:
            raise ValueError('Expect metric_name to be "r2" or "mae" or "rmse" '
                             'or "roc_auc_score", got {}'.format(metric_name))
