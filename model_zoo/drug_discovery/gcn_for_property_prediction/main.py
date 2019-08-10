import dgl
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from dgl.data import Tox21
from dgl.data.utils import download, _get_dgl_url
from model import GCNClassifier

def setup_device(args):
    cuda_available = torch.cuda.is_available()
    if cuda_available and (not args['cuda']):
        print('You may use gpu for a better speed.')
    if (not cuda_available) and args['cuda']:
        print('Failed to use gpu as it is not available.')

    args['cuda'] = args['cuda'] and cuda_available
    if args['cuda']:
        args['device'] = 'cuda:0'
    else:
        args['device'] = 'cpu'
    return args

def setup(args):
    args = setup_device(args)

    # Setup random seed
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if args['cuda']:
        torch.cuda.manual_seed(args['seed'])
    return args

def save_checkpoint(model, optimizer, checkpoint_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print('checkpoint saved')

def load_checkpoint(model, checkpoint_path, load_optimizer, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def load_pretrained_model(model):
    print('Loading pretrained model...')
    url_to_pretrained = _get_dgl_url('pre_trained/gcn_tox21.pth')
    local_pretrained_path = 'pre_trained.pth'
    download(url_to_pretrained, path=local_pretrained_path)
    load_checkpoint(model, local_pretrained_path, load_optimizer=False)

def roc_auc_averaged_over_tasks(w, y_pred, y_true):
    y_pred = torch.sigmoid(y_pred)
    n_tasks = y_true.shape[1]
    total_score = 0
    for task in range(n_tasks):
        task_w = w[:, task]
        task_y_true = y_true[:, task][task_w != 0].cpu().numpy()
        task_y_pred = y_pred[:, task][task_w != 0].cpu().detach().numpy()
        total_score += roc_auc_score(task_y_true, task_y_pred)
    return total_score / n_tasks

def run_a_train_epoch(device, num_epochs, epoch, model, loss_criterion,
                      metric_criterion, optimizer, data_loader, atom_data_field):
    model.train()
    print('Start training')
    w = []
    y_pred = []
    y_true = []
    for batch_id, batch_data in enumerate(data_loader):
        bg, labels, weights = batch_data
        atom_feats = bg.ndata.pop(atom_data_field)
        atom_feats, labels = atom_feats.to(device), labels.to(device)
        logits = model(atom_feats, bg)
        loss = (loss_criterion(logits, labels) * (weights != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
            epoch + 1, num_epochs, batch_id + 1, len(data_loader), loss.item()))

        w.append(weights)
        y_pred.append(logits)
        y_true.append(labels)
    train_score = metric_criterion(torch.cat(w, dim=0),
                                   torch.cat(y_pred, dim=0),
                                   torch.cat(y_true, dim=0))
    print('epoch {:d}/{:d}, training roc-auc score {:.4f}'.format(
        epoch + 1, num_epochs, train_score))

def run_an_eval_epoch(device, model, metric_criterion, data_loader, atom_data_field):
    model.eval()
    w = []
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            bg, labels, weights = batch_data
            atom_feats = bg.ndata.pop(atom_data_field)
            atom_feats, labels = atom_feats.to(device), labels.to(device)
            logits = model(atom_feats, bg)
            w.append(weights)
            y_pred.append(logits)
            y_true.append(labels)
    eval_score = metric_criterion(torch.cat(w, dim=0),
                                  torch.cat(y_pred, dim=0),
                                  torch.cat(y_true, dim=0))
    return eval_score

def collate(data):
    graphs, labels, weights = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    return bg, torch.stack(labels, dim=0), torch.stack(weights, dim=0)

def train(args, dataset, model):
    train_loader = DataLoader(dataset.train_set, batch_size=args['batch_size'],
                              collate_fn=collate)
    val_loader = DataLoader(dataset.val_set, batch_size=len(dataset.val_set),
                            collate_fn=collate)

    criterion = BCEWithLogitsLoss(pos_weight=torch.tensor(dataset.task_pos_weights), reduction='none')
    optimizer = Adam(model.parameters(), lr=args['lr'])
    best_val_score = 0

    for epoch in range(args['num_epochs']):
        run_a_train_epoch(args['device'], args['num_epochs'], epoch, model, criterion,
                          roc_auc_averaged_over_tasks, optimizer, train_loader, dataset.atom_data_field)
        val_score = run_an_eval_epoch(args['device'], model, roc_auc_averaged_over_tasks,
                                      val_loader, dataset.atom_data_field)
        if val_score > best_val_score:
            best_val_score = val_score
            save_checkpoint(model, optimizer, args['checkpoint_path'])
        print('epoch {:d}/{:d}, validation roc-auc score {:.4f}, best validation roc-auc score {:.4f}'.format(
            epoch + 1, args['num_epochs'], val_score, best_val_score))

    load_checkpoint(model, args['checkpoint_path'], load_optimizer=False, optimizer=None)

def test(args, dataset, model):
    test_loader = DataLoader(dataset.test_set, batch_size=len(dataset.test_set),
                             collate_fn=collate)
    # Evaluation on the test set
    test_score = run_an_eval_epoch(args['device'], model, roc_auc_averaged_over_tasks,
                                   test_loader, dataset.atom_data_field)
    print('test roc-auc score {:.4f}'.format(test_score))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='GCN for Tox21')
    parser.add_argument('-c', '--cuda', action='store_true',
                        help='Whether to use cuda')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='Random seed to use')
    parser.add_argument('-ne', '--num-epochs', type=int, default=10,
                        help='Number of epochs for training')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='Batch size for prediction')
    parser.add_argument('-lr', '--lr', type=float, default=0.001,
                        help='Learning rate for gradient descent')
    parser.add_argument('-d', '--dropout', type=float, default=0.25,
                        help='Dropout rate')
    parser.add_argument('-p', '--pre-trained', action='store_true',
                        help='Whether to skip training and use a pre-trained model')
    parser.add_argument('-cp', '--checkpoint-path', type=str, default='pretrained_model.pth',
                        help='Path to save and reload model checkpoint')
    args = parser.parse_args().__dict__
    args = setup(args)

    dataset = Tox21()
    model = GCNClassifier(in_feats=dataset.feat_size,
                          gcn_hidden_feats=[64, 64],
                          n_tasks=dataset.num_tasks,
                          classifier_hidden_feats=64,
                          dropout=args['dropout'])
    model = model.to(args['device'])

    if args['pre_trained']:
        load_pretrained_model(model)
    else:
        train(args, dataset, model)
    test(args, dataset, model)
