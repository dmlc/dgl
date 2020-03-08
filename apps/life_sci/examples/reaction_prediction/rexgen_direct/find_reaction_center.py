from dgllife.data import USPTO
from dgllife.model import WLNReactionCenter
from torch.nn import BCEWithLogitsLoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils import setup, load_data, collate

def perform_prediction(device, model, mol_graphs, complete_graphs):
    node_feats = mol_graphs.ndata.pop('hv').to(device)
    edge_feats = mol_graphs.edata.pop('he').to(device)
    node_pair_feats = complete_graphs.edata.pop('feats').to(device)

    return model(mol_graphs, complete_graphs, node_feats, edge_feats, node_pair_feats)

def main(args):
    setup(args)
    train_set = USPTO('train')
    train_loader = DataLoader(train_set, batch_size=args['batch_size'],
                              collate_fn=collate, shuffle=True)

    model = WLNReactionCenter(node_in_feats=args['node_in_feats'],
                              edge_in_feats=args['edge_in_feats'],
                              node_pair_in_feats=args['node_pair_in_feats'],
                              node_out_feats=args['node_out_feats'],
                              n_layers=args['n_layers'],
                              n_tasks=args['n_tasks']).to(args['device'])
    criterion = BCEWithLogitsLoss(reduction='sum')
    optimizer = Adam(model.parameters(), lr=args['lr'])

    for epoch in range(args['num_epochs']):
        model.train()
        for batch_id, batch_data in enumerate(train_loader):
            batch_reactions, batch_graph_edits, batch_mols, batch_mol_graphs, \
            batch_complete_graphs, batch_atom_pair_labels = batch_data
            labels = batch_atom_pair_labels.to(args['device'])
            pred = perform_prediction(args['device'], model,
                                      batch_mol_graphs, batch_complete_graphs)
            loss = criterion(pred, labels) / len(batch_reactions)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), args['max_norm'])
            optimizer.step()

if __name__ == '__main__':
    from argparse import ArgumentParser

    from configure import reaction_center_config

    parser = ArgumentParser(description='Reaction Center Identification')
    parser.add_argument('-r', '--result-path', type=str, default='results',
                        help='Path to training results')
    args = parser.parse_args().__dict__
    args.update(reaction_center_config)

    main(args)
