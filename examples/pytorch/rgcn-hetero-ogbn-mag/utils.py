from copy import deepcopy
from typing import Dict, List, Tuple, Union

import dgl
import torch
import torch.nn as nn
from ogb.nodeproppred import DglNodePropPredDataset


class Callback:
    def __init__(
        self,
        patience: int,
        monitor: str,
    ) -> None:
        self._patience = patience
        self._monitor = monitor
        self._lookback = 0
        self._best_epoch = None
        self._train_times = []
        self._valid_times = []
        self._train_losses = []
        self._valid_losses = []
        self._train_accuracies = []
        self._valid_accuracies = []
        self._model_parameters = {}

    @property
    def best_epoch(self) -> int:
        return self._best_epoch + 1

    @property
    def train_times(self) -> List[float]:
        return self._train_times

    @property
    def valid_times(self) -> List[float]:
        return self._valid_times

    @property
    def train_losses(self) -> List[float]:
        return self._train_losses

    @property
    def valid_losses(self) -> List[float]:
        return self._valid_losses

    @property
    def train_accuracies(self) -> List[float]:
        return self._train_accuracies

    @property
    def valid_accuracies(self) -> List[float]:
        return self._valid_accuracies

    @property
    def best_epoch_training_time(self) -> float:
        return sum(self._train_times[:self._best_epoch])

    @property
    def best_epoch_train_loss(self) -> float:
        return self._train_losses[self._best_epoch]

    @property
    def best_epoch_valid_loss(self) -> float:
        return self._valid_losses[self._best_epoch]

    @property
    def best_epoch_train_accuracy(self) -> float:
        return self._train_accuracies[self._best_epoch]

    @property
    def best_epoch_valid_accuracy(self) -> float:
        return self._valid_accuracies[self._best_epoch]

    @property
    def best_epoch_model_parameters(
            self) -> Union[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
        return self._model_parameters

    @property
    def should_stop(self) -> bool:
        return self._lookback >= self._patience

    def create(
        self,
        epoch: int,
        train_time: float,
        valid_time: float,
        train_loss: float,
        valid_loss: float,
        train_accuracy: float,
        valid_accuracy: float,
        model: Union[nn.Module, Dict[str, nn.Module]],
    ) -> None:
        self._train_times.append(train_time)
        self._valid_times.append(valid_time)
        self._train_losses.append(train_loss)
        self._valid_losses.append(valid_loss)
        self._train_accuracies.append(train_accuracy)
        self._valid_accuracies.append(valid_accuracy)

        best_epoch = False

        if self._best_epoch is None:
            best_epoch = True
        elif self._monitor == 'loss':
            if valid_loss < self._valid_losses[self._best_epoch]:
                best_epoch = True
        elif self._monitor == 'accuracy':
            if valid_accuracy > self._valid_accuracies[self._best_epoch]:
                best_epoch = True

        if best_epoch:
            self._best_epoch = epoch

            if isinstance(model, dict):
                for name, current_model in model.items():
                    self._model_parameters[name] = deepcopy(
                        current_model.to('cpu').state_dict())
            else:
                self._model_parameters = deepcopy(model.to('cpu').state_dict())

            self._lookback = 0
        else:
            self._lookback += 1


class OGBDataset:
    def __init__(
        self,
        g: Union[dgl.DGLGraph, dgl.DGLHeteroGraph],
        num_labels: int,
        predict_category: str = None,
    ) -> None:
        self._g = g
        self._num_labels = num_labels
        self._predict_category = predict_category

    @property
    def num_labels(self) -> int:
        return self._num_labels

    @property
    def num_classes(self) -> int:
        return self._num_labels

    @property
    def predict_category(self) -> str:
        return self._predict_category

    def __getitem__(self, idx: int) -> Union[dgl.DGLGraph, dgl.DGLHeteroGraph]:
        return self._g


def load_ogbn_mag(root: str = None) -> OGBDataset:
    dataset = DglNodePropPredDataset(name='ogbn-mag', root=root)

    split_idx = dataset.get_idx_split()

    train_idx = split_idx['train']['paper']
    valid_idx = split_idx['valid']['paper']
    test_idx = split_idx['test']['paper']

    hg_original, labels = dataset[0]

    labels = labels['paper'].squeeze()
    num_labels = dataset.num_classes

    subgraphs = {}

    for etype in hg_original.canonical_etypes:
        src, dst = hg_original.all_edges(etype=etype)

        subgraphs[etype] = (src, dst)
        subgraphs[(etype[2], f'rev-{etype[1]}', etype[0])] = (dst, src)

    hg = dgl.heterograph(subgraphs)

    hg.nodes['paper'].data['feat'] = hg_original.nodes['paper'].data['feat']
    hg.nodes['paper'].data['labels'] = labels

    train_mask = torch.zeros((hg.num_nodes('paper'),), dtype=torch.bool)
    train_mask[train_idx] = True
    valid_mask = torch.zeros((hg.num_nodes('paper'),), dtype=torch.bool)
    valid_mask[valid_idx] = True
    test_mask = torch.zeros((hg.num_nodes('paper'),), dtype=torch.bool)
    test_mask[test_idx] = True

    hg.nodes['paper'].data['train_mask'] = train_mask
    hg.nodes['paper'].data['valid_mask'] = valid_mask
    hg.nodes['paper'].data['test_mask'] = test_mask

    ogb_dataset = OGBDataset(hg, num_labels, 'paper')

    return ogb_dataset


def process_dataset(
    name: str,
    root: str = None,
) -> Tuple[OGBDataset, dgl.DGLHeteroGraph, torch.Tensor]:
    if root is None:
        root = 'datasets'

    if name == 'ogbn-mag':
        dataset = load_ogbn_mag(root=root)

    g = dataset[0]

    predict_category = dataset.predict_category

    train_idx = torch.nonzero(
        g.nodes[predict_category].data['train_mask'], as_tuple=True)[0]
    valid_idx = torch.nonzero(
        g.nodes[predict_category].data['valid_mask'], as_tuple=True)[0]
    test_idx = torch.nonzero(
        g.nodes[predict_category].data['test_mask'], as_tuple=True)[0]

    return dataset, g, train_idx, valid_idx, test_idx
