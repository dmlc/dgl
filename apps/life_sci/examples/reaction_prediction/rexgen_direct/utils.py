import dgl
import errno
import numpy as np
import os
import random
import torch

def mkdir_p(path):
    """Create a folder for the given path.
    Parameters
    ----------
    path: str
        Folder to create
    """
    try:
        os.makedirs(path)
        print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory {} already exists.'.format(path))
        else:
            raise

def setup(args, seed=0):
    """Setup for the experiment:

    1. Decide whether to use CPU or GPU for training
    2. Fix random seed for python, NumPy and PyTorch.

    Parameters
    ----------
    seed : int
        Random seed to use.

    Returns
    -------
    args
        Updated configuration
    """
    if torch.cuda.is_available():
        args['device'] = 'cuda:0'
    else:
        args['device'] = 'cpu'

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    mkdir_p(args['result_path'])

    return args

def collate(data):
    """Collate multiple datapoints

    Parameters
    ----------
    data : list of 7-tuples
        Each tuple is for a single datapoint, consisting of
        a reaction, graph edits in the reaction, an RDKit molecule instance for all reactants,
        a DGLGraph for all reactants, a complete graph for all reactants, the features for each
        pair of atoms and the labels for each pair of atoms.

    Returns
    -------
    reactions : list of str
        List of reactions.
    graph_edits : list of str
        List of graph edits in the reactions.
    mols : list of rdkit.Chem.rdchem.Mol
        List of RDKit molecule instances for the reactants.
    batch_mol_graphs : DGLGraph
        DGLGraph for a batch of molecular graphs.
    batch_complete_graphs : DGLGraph
        DGLGraph for a batch of complete graphs.
    batch_atom_pair_labels : float32 tensor of shape (V, 10)
        Labels of atom pairs in the batch of graphs.
    """
    reactions, graph_edits, mols, mol_graphs, complete_graphs, \
    atom_pair_feats, atom_pair_labels = map(list, zip(*data))

    batch_mol_graphs = dgl.batch(mol_graphs)
    batch_mol_graphs.set_n_initializer(dgl.init.zero_initializer)
    batch_mol_graphs.set_e_initializer(dgl.init.zero_initializer)

    batch_complete_graphs = dgl.batch(complete_graphs)
    batch_complete_graphs.set_n_initializer(dgl.init.zero_initializer)
    batch_complete_graphs.set_e_initializer(dgl.init.zero_initializer)
    batch_complete_graphs.edata['feats'] = torch.cat(atom_pair_feats, dim=0)

    batch_atom_pair_labels = torch.cat(atom_pair_labels, dim=0)

    return reactions, graph_edits, mols, batch_mol_graphs, \
           batch_complete_graphs, batch_atom_pair_labels
