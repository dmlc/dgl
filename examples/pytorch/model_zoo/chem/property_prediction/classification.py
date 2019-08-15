from dgl.data import Tox21
from dgl.data.utils import split_dataset
from dgl import model_zoo
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils import Meter, EarlyStopping, collate_molgraphs

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 100

    # Interchangeable with other Dataset
    dataset = Tox21()
    atom_data_field = 'h'

    trainset, valset, testset = split_dataset(dataset, [0.8, 0.1, 0.1])
    train_loader = DataLoader(
        trainset, batch_size=batch_size, collate_fn=collate_molgraphs)
    val_loader = DataLoader(
        valset, batch_size=batch_size, collate_fn=collate_molgraphs)
    test_loader = DataLoader(
        testset, batch_size=batch_size, collate_fn=collate_molgraphs)

    # Interchangeable with other model in model zoo
    model = model_zoo.chem.GCNClassifier(in_feats=74,
                                         gcn_hidden_feats=[64, 64],
                                         n_tasks=dataset.n_tasks)

    model.to(device)

    if args.pre_trained:
        num_epochs = 0
        model_zoo.load_pretrained('GCN_Tox21')
    else:
        loss_criterion = BCEWithLogitsLoss(pos_weight=torch.tensor(
            dataset.task_pos_weights).to(device), reduction='none')
        optimizer = Adam(model.parameters(), lr=learning_rate)
        stopper = EarlyStopping(patience=10)

    for epoch in range(num_epochs):
        model.train()
        print('Start training')
        train_meter = Meter()
        for batch_id, batch_data in enumerate(train_loader):
            smiles, bg, labels, mask = batch_data
            atom_feats = bg.ndata.pop(atom_data_field)
            atom_feats, labels, mask = atom_feats.to(device), labels.to(device), mask.to(device)
            logits = model(atom_feats, bg)
            # Mask non-existing labels
            loss = (loss_criterion(logits, labels)
                    * (mask != 0).float()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
                epoch + 1, num_epochs, batch_id + 1, len(train_loader), loss.item()))
            train_meter.update(logits, labels, mask)
        train_roc_auc = train_meter.roc_auc_averaged_over_tasks()
        print('epoch {:d}/{:d}, training roc-auc score {:.4f}'.format(
            epoch + 1, num_epochs, train_roc_auc))
        
        val_meter = Meter()
        model.eval()
        with torch.no_grad():
            for batch_id, batch_data in enumerate(val_loader):
                smiles, bg, labels, mask = batch_data
                atom_feats = bg.ndata.pop(atom_data_field)
                atom_feats, labels = atom_feats.to(device), labels.to(device)
                logits = model(atom_feats, bg)
                val_meter.update(logits, labels, mask)
        
        val_roc_auc = val_meter.roc_auc_averaged_over_tasks()
        if stopper.step(val_roc_auc, model):
            break

        print('epoch {:d}/{:d}, validation roc-auc score {:.4f}, best validation roc-auc score {:.4f}'.format(
            epoch + 1, num_epochs, val_roc_auc, stopper.best_score))

    test_meter = Meter()
    model.eval()
    for batch_id, batch_data in enumerate(test_loader):
        smiles, bg, labels, mask = batch_data
        atom_feats = bg.ndata.pop(atom_data_field)
        atom_feats, labels = atom_feats.to(device), labels.to(device)
        logits = model(atom_feats, bg)
        test_meter.update(logits, labels, mask)
    print('test roc-auc score {:.4f}'.format(test_meter.roc_auc_averaged_over_tasks()))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Molecule Classification')
    parser.add_argument('-p', '--pre-trained', action='store_true',
                        help='Whether to skip training and use a pre-trained model')
    args = parser.parse_args()

    main(args)
