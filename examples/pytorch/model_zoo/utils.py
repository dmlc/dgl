import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import dgl

class Meter():
    def __init__(self, mode=None):
        self.mode = mode
        self.mask = []
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true, mask):
        self.y_pred.append(y_pred)
        self.y_true.append(y_true)
        self.mask.append(mask)

    def roc_auc_averaged_over_tasks(self):

        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        total_score = 0
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].cpu().numpy()
            task_y_pred = y_pred[:, task][task_w != 0].cpu().detach().numpy()
            total_score += roc_auc_score(task_y_true, task_y_pred)
        return total_score / n_tasks


class EarlyStopping:
    def __init__(self, patience=10, filename="es_checkpoint.pt"):
        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        '''Load early stopping model'''
        model.load_state_dict(torch.load(self.filename))


def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader
    Parameters
    ----------
    data : list of 3-tuples
        Each tuple is for a single datapoint, consisting of
        a DGLGraph, all-task labels and all-task weights
    Returns
    -------
    smiles : list
        List of smiles
    bg : BatchedDGLGraph
        Batched DGLGraphs
    labels : Tensor of dtype float32 and shape (B, 12)
        Batched datapoint labels. B is len(data) and
        12 is the number of total tasks.
    weights : Tensor of dtype float32 and shape (B, 12)
        Batched datapoint weights.
    """
    smiles, graphs, labels, mask = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)
    mask = torch.stack(mask, dim=0)
    return smiles, bg, labels, mask
