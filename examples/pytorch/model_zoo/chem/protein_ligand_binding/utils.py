import datetime
import dgl
import numpy as np
import random
import torch
import torch.nn.functional as F

from dgl import model_zoo
from dgl.data.chem import PDBBind, RandomSplitter
from sklearn.metrics import r2_score

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_dataset(args):
    assert args['dataset'] in ['PDBBind'], 'Unexpected dataset {}'.format(args['dataset'])
    if args['dataset'] == 'PDBBind':
        dataset = PDBBind(subset=args['subset'], load_binding_pocket=args['load_binding_pocket'])
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, rac_train=args['frac_train'], frac_val=args['frac_val'],
            frac_test=args['frac_test'], random_state=args['random_seed'])

    return dataset, train_set, val_set, test_set

def collate(data):
    """"""
    indices, protein_mols, ligand_mols, graphs, labels = map(list, zip(*data))
    bg = dgl.batch_hetero(graphs)
    for nty in bg.ntypes:
        bg.set_n_initializer(dgl.init.zero_initializer, ntype=nty)
    for ety in bg.canonical_etypes:
        bg.set_e_initializer(dgl.init.zero_initializer, etype=ety)
    labels = torch.stack(labels, dim=0)

    return indices, protein_mols, ligand_mols, bg, labels

def load_model(args):
    """"""
    assert args['model'] in ['ACNN'], 'Unexpected model {}'.format(args['model'])
    if args['model'] == 'ACNN':
        model = model_zoo.chem.ACNN(hidden_sizes=args['hidden_sizes'],
                                    features_to_use=args['atomic_numbers_considered'],
                                    radial=args['radial'])

    return model

class EarlyStopping(object):
    """Early stop performing
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
    patience : int
        Number of epochs to wait before early stop
        if the metric stops getting improved
    filename : str or None
        Filename for storing the model checkpoint
    """
    def __init__(self, mode='higher', patience=10, filename=None):
        if filename is None:
            dt = datetime.datetime.now()
            filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
                dt.date(), dt.hour, dt.minute, dt.second)

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        return (score > prev_best_score)

    def _check_lower(self, score, prev_best_score):
        return (score < prev_best_score)

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)

    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])

class Meter(object):
    """Track and summarize model performance on a dataset for (multi-label) prediction.

    Parameters
    ----------
    torch.float32 tensor of shape (T)
        Mean of existing training labels across tasks, T for the number of tasks
    torch.float32 tensor of shape (T)
        Std of existing training labels across tasks, T for the number of tasks
    """
    def __init__(self):
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true):
        """Update for the result of an iteration

        Parameters
        ----------
        y_pred : float32 tensor
            Predicted molecule labels with shape (B, T),
            B for batch size and T for the number of tasks
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())

    def r2(self):
        """Compute R2 (coefficient of determination)

        Returns
        -------
        float
        """
        y_pred = torch.cat(self.y_pred, dim=0)
        # To compensate for the imbalance between labels during training,
        # we normalize the ground truth labels with training mean and std.
        # We need to undo that for evaluation.
        y_true = torch.cat(self.y_true, dim=0)

        return r2_score(y_true[:, 0].numpy(), y_pred[:, 0].numpy())

    def l1(self):
        """Compute l1 loss for each task.

        Returns
        -------
        list of float
            l1 loss for all tasks
        reduction : str
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        """
        y_pred = torch.cat(self.y_pred, dim=0)
        # To compensate for the imbalance between labels during training,
        # we normalize the ground truth labels with training mean and std.
        # We need to undo that for evaluation.
        y_pred = y_pred * self.std + self.mean
        y_true = torch.cat(self.y_true, dim=0)

        return F.l1_loss(y_true, y_pred).data.item()

    def rmse(self):
        """
        Compute RMSE

        Returns
        -------
        float
        """
        y_pred = torch.cat(self.y_pred, dim=0)
        # To compensate for the imbalance between labels during training,
        # we normalize the ground truth labels with training mean and std.
        # We need to undo that for evaluation.
        y_pred = y_pred * self.std + self.mean
        y_true = torch.cat(self.y_true, dim=0)

        return np.sqrt(F.mse_loss(y_pred, y_true).cpu().item())

    def compute_metric(self, metric_name):
        """Compute metric for each task.

        Parameters
        ----------
        metric_name : str
            Name for the metric to compute.

        Returns
        -------
        list of float
            Metric value for each task
        """
        assert metric_name in ['r2', 'l1', 'rmse'], \
            'Expect metric name to be "r2", "l1" or "rmse", got {}'.format(metric_name)
        if metric_name == 'r2':
            return self.r2()
        if metric_name == 'l1':
            return self.l1()
        if metric_name == 'rmse':
            return self.rmse()
