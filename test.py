import dgl
import numpy as np
import random
import torch
import torch.nn as nn

from dgllife.data import Tox21
from dgllife.model import GINPredictor, load_pretrained
from dgllife.utils import smiles_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer, \
    ScaffoldSplitter, Meter
from torch.optim import Adam
from torch.utils.data import DataLoader

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

def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.

    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally
        a binary mask indicating the existence of labels.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels. If binary masks are not
        provided, return a tensor with ones.
    """
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles, bg, labels, masks

def predict(args, model, bg):
    atom_types = bg.ndata.pop('atomic_number').to(args['device'])
    chirality_types = bg.ndata.pop('chirality_type').to(args['device'])
    bond_types = bg.edata.pop('bond_type').to(args['device'])
    bond_direction_types = bg.edata.pop('bond_direction_type').to(args['device'])

    return model(bg, [atom_types, chirality_types], [bond_types, bond_direction_types])

def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        labels, masks = labels.to(args['device']), masks.to(args['device'])
        logits = predict(args, model, bg)
        # Mask non-existing labels
        loss = (loss_criterion(logits, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
            epoch + 1, args['epochs'], batch_id + 1, len(data_loader), loss.item()))
        train_meter.update(logits, labels, masks)
    train_score = np.mean(train_meter.compute_metric(args['metric_name']))
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, args['epochs'], args['metric_name'], train_score))

def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.to(args['device'])
            logits = predict(args, model, bg)
            eval_meter.update(logits, labels, masks)
    return np.mean(eval_meter.compute_metric(args['metric_name']))

def main(args):
    args['device'] = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed()

    dataset = Tox21(smiles_to_graph=smiles_to_bigraph,
                    node_featurizer=PretrainAtomFeaturizer(),
                    edge_featurizer=PretrainBondFeaturizer())
    train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(dataset)
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    val_loader = DataLoader(val_set, batch_size=args['batch_size'], shuffle=False,
                            collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False,
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    model = GINPredictor(num_node_emb_list=args['num_node_emb_list'],
                         num_edge_emb_list=args['num_edge_emb_list'],
                         num_layers=args['num_layers'],
                         emb_dim=args['emb_dim'],
                         JK=args['JK'],
                         dropout=args['dropout'],
                         readout=args['readout'],
                         n_tasks=args['n_tasks'])
    model.gnn = load_pretrained('gin_supervised_{}'.format(args['unsup']))
    model.to(args['device'])
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = Adam(model.parameters(), lr=args['lr'], weight_decay=args['decay'])

    for epoch in range(1, args['epochs'] + 1):
        run_a_train_epoch(args, epoch, model, train_loader, criterion, optimizer)

        # Validation and early stop
        val_score = run_an_eval_epoch(args, model, val_loader)
        print('epoch {:d}/{:d}, validation {} {:.4f}'.format(
            epoch + 1, args['epochs'], args['metric_name'],
            val_score, args['metric_name']))

    test_score = run_an_eval_epoch(args, model, test_loader)
    print('test {} {:.4f}'.format(args['metric_name'], test_score))

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-u', '--unsup', type=str,
                        choices=['contextpred', 'edgepred', 'infomax', 'masking'])
    args = parser.parse_args().__dict__
    args.update({
        'batch_size': 32,
        'epochs': 100,
        'lr': 0.001,
        'decay': 0.,
        'num_layers': 5,
        'emb_dim': 300,
        'dropout': 0.5,
        'readout': 'mean',
        'JK': 'last',
        'num_workers': 4,
        'n_tasks': 12,
        'num_node_emb_list': [120, 3],
        'num_edge_emb_list': [6, 3],
        'metric_name': 'roc_auc_score'
    })
    main(args)
