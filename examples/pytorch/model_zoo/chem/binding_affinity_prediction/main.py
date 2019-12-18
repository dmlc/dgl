import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from utils import set_random_seed, load_dataset, collate, load_model, Meter

def update_msg_from_scores(msg, scores):
    for metric, score in scores.items():
        msg += ', {} {:.4f}'.format(metric, score)
    return msg

def run_a_train_epoch(args, epoch, model, data_loader,
                      loss_criterion, optimizer):
    model.train()
    train_meter = Meter(args['train_mean'], args['train_std'])
    epoch_loss = 0
    for batch_id, batch_data in enumerate(data_loader):
        indices, ligand_mols, protein_mols, bg, labels = batch_data
        labels, bg = labels.to(args['device']), bg.to(args['device'])
        prediction = model(bg)
        loss = loss_criterion(prediction, (labels - args['train_mean']) / args['train_std'])
        epoch_loss += loss.data.item() * len(indices)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels)
    avg_loss = epoch_loss / len(data_loader.dataset)
    total_scores = {metric: train_meter.compute_metric(metric) for metric in args['metrics']}
    msg = 'epoch {:d}/{:d}, training | loss {:.4f}'.format(
        epoch + 1, args['num_epochs'], avg_loss)
    msg = update_msg_from_scores(msg, total_scores)
    print(msg)

def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter(args['train_mean'], args['train_std'])
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            indices, ligand_mols, protein_mols, bg, labels = batch_data
            labels, bg = labels.to(args['device']), bg.to(args['device'])
            prediction = model(bg)
            eval_meter.update(prediction, labels)
    total_scores = {metric: eval_meter.compute_metric(metric) for metric in args['metrics']}
    return total_scores

def main(args):
    args['device'] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(args['random_seed'])

    dataset, train_set, test_set = load_dataset(args)
    args['train_mean'] = train_set.labels_mean.to(args['device'])
    args['train_std'] = train_set.labels_std.to(args['device'])
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['batch_size'],
                              shuffle=False,
                              collate_fn=collate)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=args['batch_size'],
                             shuffle=True,
                             collate_fn=collate)

    model = load_model(args)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    model.to(args['device'])

    for epoch in range(args['num_epochs']):
        run_a_train_epoch(args, epoch, model, train_loader, loss_fn, optimizer)

    test_scores = run_an_eval_epoch(args, model, test_loader)
    test_msg = update_msg_from_scores('test results', test_scores)
    print(test_msg)

if __name__ == '__main__':
    import argparse

    from configure import get_exp_configure

    parser = argparse.ArgumentParser(description='Protein-Ligand Binding Affinity Prediction')
    parser.add_argument('-m', '--model', type=str, choices=['ACNN'],
                        help='Model to use')
    parser.add_argument('-d', '--dataset', type=str,
                        choices=['PDBBind_core_pocket_random', 'PDBBind_core_pocket_scaffold',
                                 'PDBBind_core_pocket_stratified', 'PDBBind_core_pocket_temporal',
                                 'PDBBind_refined_pocket_random', 'PDBBind_refined_pocket_scaffold',
                                 'PDBBind_refined_pocket_stratified', 'PDBBind_refined_pocket_temporal'],
                        help='Dataset to use')

    args = parser.parse_args().__dict__
    args['exp'] = '_'.join([args['model'], args['dataset']])
    args.update(get_exp_configure(args['exp']))

    main(args)
