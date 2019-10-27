import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dgl import model_zoo

from utils import Meter, set_random_seed, collate_molgraphs, EarlyStopping, \
    load_dataset_for_regression, load_model

def regress(args, model, bg):
    if args['model'] == 'MPNN':
        h = bg.ndata.pop('n_feat')
        e = bg.edata.pop('e_feat')
        h, e = h.to(args['device']), e.to(args['device'])
        return model(bg, h, e)
    elif args['model'] in ['SCHNET', 'MGCN']:
        node_types = bg.ndata.pop('node_type')
        edge_distances = bg.edata.pop('distance')
        node_types, edge_distances = node_types.to(args['device']), \
                                     edge_distances.to(args['device'])
        return model(bg, node_types, edge_distances)
    else:
        atom_feats, bond_feats = bg.ndata.pop('hv'), bg.edata.pop('he')
        atom_feats, bond_feats = atom_feats.to(args['device']), bond_feats.to(args['device'])
        return model(bg, atom_feats, bond_feats)

def run_a_train_epoch(args, epoch, model, data_loader,
                      loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        labels, masks = labels.to(args['device']), masks.to(args['device'])
        prediction = regress(args, model, bg)
        loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels, masks)
    total_score = np.mean(train_meter.compute_metric(args['metric_name']))
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, args['num_epochs'], args['metric_name'], total_score))

def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.to(args['device'])
            prediction = regress(args, model, bg)
            eval_meter.update(prediction, labels, masks)
        total_score = np.mean(eval_meter.compute_metric(args['metric_name']))
    return total_score

def main(args):
    args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    set_random_seed()

    train_set, val_set, test_set = load_dataset_for_regression(args)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['batch_size'],
                              shuffle=True,
                              collate_fn=collate_molgraphs)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=args['batch_size'],
                            shuffle=True,
                            collate_fn=collate_molgraphs)
    if test_set is not None:
        test_loader = DataLoader(dataset=test_set,
                                 batch_size=args['batch_size'],
                                 collate_fn=collate_molgraphs)

    if args['pre_trained']:
        args['num_epochs'] = 0
        model = model_zoo.chem.load_pretrained(args['exp'])
    else:
        model = load_model(args)
        if args['model'] in ['SCHNET', 'MGCN']:
            model.set_mean_std(train_set.mean, train_set.std, args['device'])
        loss_fn = nn.MSELoss(reduction='none')
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        stopper = EarlyStopping(mode='lower', patience=args['patience'])
    model.to(args['device'])

    for epoch in range(args['num_epochs']):
        # Train
        run_a_train_epoch(args, epoch, model, train_loader, loss_fn, optimizer)

        # Validation and early stop
        val_score = run_an_eval_epoch(args, model, val_loader)
        early_stop = stopper.step(val_score, model)
        print('epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['metric_name'], val_score,
            args['metric_name'], stopper.best_score))

        if early_stop:
            break

    if test_set is not None:
        if not args['pre_trained']:
            stopper.load_checkpoint(model)
        test_score = run_an_eval_epoch(args, model, test_loader)
        print('test {} {:.4f}'.format(args['metric_name'], test_score))

if __name__ == "__main__":
    import argparse

    from configure import get_exp_configure

    parser = argparse.ArgumentParser(description='Molecule Regression')
    parser.add_argument('-m', '--model', type=str,
                        choices=['MPNN', 'SCHNET', 'MGCN', 'AttentiveFP'],
                        help='Model to use')
    parser.add_argument('-d', '--dataset', type=str, choices=['Alchemy', 'Aromaticity'],
                        help='Dataset to use')
    parser.add_argument('-p', '--pre-trained', action='store_true',
                        help='Whether to skip training and use a pre-trained model')
    args = parser.parse_args().__dict__
    args['exp'] = '_'.join([args['model'], args['dataset']])
    args.update(get_exp_configure(args['exp']))

    main(args)
