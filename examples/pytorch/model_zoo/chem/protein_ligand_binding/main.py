import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from utils import set_random_seed, load_dataset, collate, load_model, EarlyStopping, Meter

def run_a_train_epoch(args, epoch, model, data_loader,
                      loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        indices, protein_mols, ligand_mols, bg, labels = batch_data
        labels, bg = labels.to(args['device']), bg.to(args['device'])
        prediction = model(bg)
        loss = loss_criterion(prediction, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels)
    total_score = train_meter.compute_metric(args['metric'])
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, args['num_epochs'], args['metric'], total_score))

def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            indices, protein_mols, ligand_mols, bg, labels = batch_data
            labels, bg = labels.to(args['device']), bg.to(args['device'])
            prediction = model(bg)
            eval_meter.update(prediction, labels)
        total_score = eval_meter.compute_metric(args['metric'])
    return total_score

def main(args):
    args['device'] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(args['seed'])

    dataset, train_set, val_set, test_set = load_dataset(args)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['batch_size'],
                              shuffle=True,
                              collate_fn=collate)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=args['batch_size'],
                            shuffle=True,
                            collate_fn=collate)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=args['batch_size'],
                             shuffle=True,
                             collate_fn=collate)
    model = load_model(args)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    stopper = EarlyStopping(mode='higher')
    model.to(args['device'])

    for epoch in range(args['num_epochs']):
        # Train
        run_a_train_epoch(args, epoch, model, train_loader, loss_fn, optimizer)

        # Validation and early stop
        val_score = run_an_eval_epoch(args, model, val_loader)
        early_stop = stopper.step(val_score, model)
        print('epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['metric'], val_score,
            args['metric'], stopper.best_score))

        if early_stop:
            break

    test_score = run_an_eval_epoch(args, model, test_loader)
    print('test {} {:.4f}'.format(args['metric'], test_score))

if __name__ == '__main__':
    import argparse

    from configure import get_exp_configure

    parser = argparse.ArgumentParser(description='Protein-Ligand Binding Affinity Prediction')
    parser.add_argument('-m', '--model', type=str, choices=['ACNN'],
                        help='Model to use')
    parser.add_argument('-d', '--dataset', type=str,
                        choices=['PDBBind_core_protein', 'PDBBind_core_pocket',
                                 'PDBBind_refined_protein', 'PDBBind_refined_pocket'],
                        help='Dataset to use')

    args = parser.parse_args().__dict__
    args['exp'] = '_'.join([args['model'], args['dataset']])
    args.update(get_exp_configure(args['exp']))

    main(args)
