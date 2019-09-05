import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dgl import model_zoo

from utils import set_random_seed, collate_molgraphs_for_regression, EarlyStopping

def regress(args, model, bg):
    if args['model'] == 'MPNN':
        h = bg.ndata.pop('n_feat')
        e = bg.edata.pop('e_feat')
        h, e = h.to(args['device']), e.to(args['device'])
        return model(bg, h, e)
    else:
        node_types = bg.ndata.pop('node_type')
        edge_distances = bg.edata.pop('distance')
        node_types, edge_distances = node_types.to(args['device']), \
                                     edge_distances.to(args['device'])
        return model(bg, node_types, edge_distances)

def run_a_train_epoch(args, epoch, model, data_loader,
                      loss_criterion, score_criterion, optimizer):
    model.train()
    total_loss, total_score = 0, 0
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels = batch_data
        labels = labels.to(args['device'])
        prediction = regress(args, model, bg)
        loss = loss_criterion(prediction, labels)
        score = score_criterion(prediction, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
        total_score += score.detach().item()
    total_loss /= len(data_loader.dataset)
    total_score /= len(data_loader.dataset)
    print('epoch {:d}/{:d}, training loss {:.4f}, training score {:.4f}'.format(
        epoch + 1, args['num_epochs'], total_loss, total_score))

def run_an_eval_epoch(args, model, data_loader, score_criterion):
    model.eval()
    total_score = 0
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels = batch_data
            labels = labels.to(args['device'])
            prediction = regress(args, model, bg)
            score = score_criterion(prediction, labels)
            total_score += score.detach().item()
        total_score /= len(data_loader.dataset)
    return total_score

def main(args):
    args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    set_random_seed()

    # Interchangeable with other datasets
    if args['dataset'] == 'Alchemy':
        from dgl.data.chem import TencentAlchemyDataset
        train_set = TencentAlchemyDataset(mode='dev')
        val_set = TencentAlchemyDataset(mode='valid')

    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['batch_size'],
                              collate_fn=collate_molgraphs_for_regression)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=args['batch_size'],
                            collate_fn=collate_molgraphs_for_regression)

    if args['model'] == 'MPNN':
        model = model_zoo.chem.MPNNModel(output_dim=args['output_dim'])
    elif args['model'] == 'SCHNET':
        model = model_zoo.chem.SchNet(norm=args['norm'], output_dim=args['output_dim'])
        model.set_mean_std(train_set.mean, train_set.std, args['device'])
    elif args['model'] == 'MGCN':
        model = model_zoo.chem.MGCNModel(norm=args['norm'], output_dim=args['output_dim'])
        model.set_mean_std(train_set.mean, train_set.std, args['device'])
    model.to(args['device'])

    loss_fn = nn.MSELoss(reduction='sum')
    score_fn = nn.L1Loss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    stopper = EarlyStopping(mode='lower', patience=args['patience'])

    for epoch in range(args['num_epochs']):
        # Train
        run_a_train_epoch(args, epoch, model, train_loader, loss_fn, score_fn, optimizer)

        # Validation and early stop
        val_score = run_an_eval_epoch(args, model, val_loader, score_fn)
        early_stop = stopper.step(val_score, model)
        print('epoch {:d}/{:d}, validation score {:.4f}, best validation score {:.4f}'.format(
            epoch + 1, args['num_epochs'], val_score, stopper.best_score))
        if early_stop:
            break

if __name__ == "__main__":
    import argparse

    from configure import get_exp_configure

    parser = argparse.ArgumentParser(description='Molecule Regression')
    parser.add_argument('-m', '--model', type=str, choices=['MPNN', 'SCHNET', 'MGCN'],
                        help='Model to use')
    parser.add_argument('-d', '--dataset', type=str, choices=['Alchemy'],
                        help='Dataset to use')
    args = parser.parse_args().__dict__
    args['exp'] = '_'.join([args['model'], args['dataset']])
    args.update(get_exp_configure(args['exp']))

    main(args)
