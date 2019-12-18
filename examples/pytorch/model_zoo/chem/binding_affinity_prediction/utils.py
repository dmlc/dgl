import dgl
import numpy as np
import random
import torch
import torch.nn.functional as F

from dgl import model_zoo
from dgl.data.chem import PDBBind, RandomSplitter, ScaffoldSplitter, SingleTaskStratifiedSplitter
from dgl.data.utils import Subset
from itertools import accumulate
from scipy.stats import pearsonr

def set_random_seed(seed=0):
    """Set random seed.

    Parameters
    ----------
    seed : int
        Random seed to use. Default to 0.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_dataset(args):
    """Load the dataset.

    Parameters
    ----------
    args : dict
        Input arguments.

    Returns
    -------
    dataset
        Full dataset.
    train_set
        Train subset of the dataset.
    val_set
        Validation subset of the dataset.
    """
    assert args['dataset'] in ['PDBBind'], 'Unexpected dataset {}'.format(args['dataset'])
    if args['dataset'] == 'PDBBind':
        dataset = PDBBind(subset=args['subset'],
                          load_binding_pocket=args['load_binding_pocket'],
                          zero_padding=True)
        # No validation set is used and frac_val = 0.
        if args['split'] == 'random':
            train_set, _, test_set = RandomSplitter.train_val_test_split(
                dataset,
                frac_train=args['frac_train'],
                frac_val=args['frac_val'],
                frac_test=args['frac_test'],
                random_state=args['random_seed'])

        elif args['split'] == 'scaffold':
            train_set, _, test_set = ScaffoldSplitter.train_val_test_split(
                dataset,
                mols=dataset.ligand_mols,
                sanitize=False,
                frac_train=args['frac_train'],
                frac_val=args['frac_val'],
                frac_test=args['frac_test'])

        elif args['split'] == 'stratified':
            train_set, _, test_set = SingleTaskStratifiedSplitter.train_val_test_split(
                dataset,
                labels=dataset.labels,
                task_id=0,
                frac_train=args['frac_train'],
                frac_val=args['frac_val'],
                frac_test=args['frac_test'],
                random_state=args['random_seed'])

        elif args['split'] == 'temporal':
            years = dataset.df['release_year'].values.astype(np.float32)
            indices = np.argsort(years).tolist()
            frac_list = np.array([args['frac_train'], args['frac_val'], args['frac_test']])
            num_data = len(dataset)
            lengths = (num_data * frac_list).astype(int)
            lengths[-1] = num_data - np.sum(lengths[:-1])
            train_set, val_set, test_set = [
                Subset(dataset, list(indices[offset - length:offset]))
                for offset, length in zip(accumulate(lengths), lengths)]

        else:
            raise ValueError('Expect the splitting method '
                             'to be "random" or "scaffold", got {}'.format(args['split']))
        train_labels = torch.stack([train_set.dataset.labels[i] for i in train_set.indices])
        train_set.labels_mean = train_labels.mean(dim=0)
        train_set.labels_std = train_labels.std(dim=0)

    return dataset, train_set, test_set

def collate(data):
    indices, ligand_mols, protein_mols, graphs, labels = map(list, zip(*data))
    bg = dgl.batch_hetero(graphs)
    for nty in bg.ntypes:
        bg.set_n_initializer(dgl.init.zero_initializer, ntype=nty)
    for ety in bg.canonical_etypes:
        bg.set_e_initializer(dgl.init.zero_initializer, etype=ety)
    labels = torch.stack(labels, dim=0)

    return indices, ligand_mols, protein_mols, bg, labels

def load_model(args):
    assert args['model'] in ['ACNN'], 'Unexpected model {}'.format(args['model'])
    if args['model'] == 'ACNN':
        model = model_zoo.chem.ACNN(hidden_sizes=args['hidden_sizes'],
                                    weight_init_stddevs=args['weight_init_stddevs'],
                                    dropouts=args['dropouts'],
                                    features_to_use=args['atomic_numbers_considered'],
                                    radial=args['radial'])

    return model

class Meter(object):
    """Track and summarize model performance on a dataset for (multi-label) prediction.

    Parameters
    ----------
    torch.float32 tensor of shape (T)
        Mean of existing training labels across tasks, T for the number of tasks
    torch.float32 tensor of shape (T)
        Std of existing training labels across tasks, T for the number of tasks
    """
    def __init__(self, mean=None, std=None):
        self.y_pred = []
        self.y_true = []

        if (type(mean) != type(None)) and (type(std) != type(None)):
            self.mean = mean.cpu()
            self.std = std.cpu()
        else:
            self.mean = None
            self.std = None

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

    def _finalize_labels_and_prediction(self):
        """Concatenate the labels and predictions.

        If normalization was performed on the labels, undo the normalization.
        """
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)

        if (self.mean is not None) and (self.std is not None):
            # To compensate for the imbalance between labels during training,
            # we normalize the ground truth labels with training mean and std.
            # We need to undo that for evaluation.
            y_pred = y_pred * self.std + self.mean

        return y_pred, y_true

    def pearson_r2(self):
        """Compute squared Pearson correlation coefficient

        Returns
        -------
        float
        """
        y_pred, y_true = self._finalize_labels_and_prediction()

        return pearsonr(y_true[:, 0].numpy(), y_pred[:, 0].numpy())[0] ** 2

    def mae(self):
        """Compute MAE

        Returns
        -------
        float
        """
        y_pred, y_true = self._finalize_labels_and_prediction()

        return F.l1_loss(y_true, y_pred).data.item()

    def rmse(self):
        """
        Compute RMSE

        Returns
        -------
        float
        """
        y_pred, y_true = self._finalize_labels_and_prediction()

        return np.sqrt(F.mse_loss(y_pred, y_true).cpu().item())

    def compute_metric(self, metric_name):
        """Compute metric

        Parameters
        ----------
        metric_name : str
            Name for the metric to compute.

        Returns
        -------
        float
            Metric value
        """
        assert metric_name in ['pearson_r2', 'mae', 'rmse'], \
            'Expect metric name to be "pearson_r2", "mae" or "rmse", got {}'.format(metric_name)
        if metric_name == 'pearson_r2':
            return self.pearson_r2()
        if metric_name == 'mae':
            return self.mae()
        if metric_name == 'rmse':
            return self.rmse()
