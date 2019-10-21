import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from dgl import model_zoo
from dgl.data.utils import split_dataset

from utils import Meter, EarlyStopping, collate_molgraphs_for_classification, set_random_seed

def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, mask = batch_data
        atom_feats = bg.ndata.pop(args['atom_data_field'])
        atom_feats, labels, mask = atom_feats.to(args['device']), \
                                   labels.to(args['device']), \
                                   mask.to(args['device'])
        logits = model(bg, atom_feats)
        # Mask non-existing labels
        loss = (loss_criterion(logits, labels) * (mask != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
            epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
        train_meter.update(logits, labels, mask)
    train_roc_auc = train_meter.roc_auc_averaged_over_tasks()
    print('epoch {:d}/{:d}, training roc-auc score {:.4f}'.format(
        epoch + 1, args['num_epochs'], train_roc_auc))

def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, mask = batch_data
            atom_feats = bg.ndata.pop(args['atom_data_field'])
            atom_feats, labels = atom_feats.to(args['device']), labels.to(args['device'])
            logits = model(bg, atom_feats)
            eval_meter.update(logits, labels, mask)
    return eval_meter.roc_auc_averaged_over_tasks()

def main(args):
    args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    set_random_seed()

    # Interchangeable with other datasets
    if args['dataset'] == 'Tox21':
        from dgl.data.chem import Tox21
        dataset = Tox21()

    trainset, valset, testset = split_dataset(dataset, args['train_val_test_split'])
    train_loader = DataLoader(trainset, batch_size=args['batch_size'],
                              collate_fn=collate_molgraphs_for_classification)
    val_loader = DataLoader(valset, batch_size=args['batch_size'],
                            collate_fn=collate_molgraphs_for_classification)
    test_loader = DataLoader(testset, batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs_for_classification)

    if args['pre_trained']:
        args['num_epochs'] = 0
        model = model_zoo.chem.load_pretrained(args['exp'])
    else:
        # Interchangeable with other models
        if args['model'] == 'GCN':
            model = model_zoo.chem.GCNClassifier(in_feats=args['in_feats'],
                                                 gcn_hidden_feats=args['gcn_hidden_feats'],
                                                 classifier_hidden_feats=args['classifier_hidden_feats'],
                                                 n_tasks=dataset.n_tasks)
        elif args['model'] == 'GAT':
            model = model_zoo.chem.GATClassifier(in_feats=args['in_feats'],
                                                 gat_hidden_feats=args['gat_hidden_feats'],
                                                 num_heads=args['num_heads'],
                                                 classifier_hidden_feats=args['classifier_hidden_feats'],
                                                 n_tasks=dataset.n_tasks)

        loss_criterion = BCEWithLogitsLoss(pos_weight=dataset.task_pos_weights.to(args['device']),
                                           reduction='none')
        optimizer = Adam(model.parameters(), lr=args['lr'])
        stopper = EarlyStopping(patience=args['patience'])
    model.to(args['device'])

    for epoch in range(args['num_epochs']):
        # Train
        run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)

        # Validation and early stop
        val_roc_auc = run_an_eval_epoch(args, model, val_loader)
        early_stop = stopper.step(val_roc_auc, model)
        print('epoch {:d}/{:d}, validation roc-auc score {:.4f}, best validation roc-auc score {:.4f}'.format(
            epoch + 1, args['num_epochs'], val_roc_auc, stopper.best_score))
        if early_stop:
            break

    if not args['pre_trained']:
        stopper.load_checkpoint(model)
    test_roc_auc = run_an_eval_epoch(args, model, test_loader)
    print('test roc-auc score {:.4f}'.format(test_roc_auc))

if __name__ == '__main__':
    import argparse

    from configure import get_exp_configure

    parser = argparse.ArgumentParser(description='Molecule Classification')
    parser.add_argument('-m', '--model', type=str, choices=['GCN', 'GAT'],
                        help='Model to use')
    parser.add_argument('-d', '--dataset', type=str, choices=['Tox21'],
                        help='Dataset to use')
    parser.add_argument('-p', '--pre-trained', action='store_true',
                        help='Whether to skip training and use a pre-trained model')
    args = parser.parse_args().__dict__
    args['exp'] = '_'.join([args['model'], args['dataset']])
    args.update(get_exp_configure(args['exp']))

    main(args)
