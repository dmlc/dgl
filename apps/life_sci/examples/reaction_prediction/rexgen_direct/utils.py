import dgl
import errno
import numpy as np
import os
import random
import torch
import torch.distributed as dist
import torch.nn as nn

from collections import defaultdict
from dgllife.data import USPTOCenter, WLNCenterDataset
from dgllife.model import load_pretrained, WLNReactionCenter
from rdkit import Chem
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

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

def set_seed(seed=0):
    """Fix random seed.

    Parameters
    ----------
    seed : int
        Random seed to use. Default to 0.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_center_subset(dataset, subset_id, num_subsets):
    """Get subset for reaction center identification.

    Parameters
    ----------
    dataset : WLNCenterDataset
        Dataset for reaction center prediction with WLN
    subset_id : int
        Index for the subset
    num_subsets : int
        Number of total subsets
    """
    if num_subsets == 1:
        return

    total_size = len(dataset)
    subset_size = total_size // num_subsets
    start = subset_id * subset_size
    end = (subset_id + 1) * subset_size

    dataset.mols = dataset.mols[start:end]
    dataset.reactions = dataset.reactions[start:end]
    dataset.graph_edits = dataset.graph_edits[start:end]
    dataset.reactant_mol_graphs = dataset.reactant_mol_graphs[start:end]
    dataset.atom_pair_features = [None for _ in range(subset_size)]
    dataset.atom_pair_labels = [None for _ in range(subset_size)]

class Optimizer(nn.Module):
    """Wrapper for optimization

    Parameters
    ----------
    model : nn.Module
        Model being trained
    lr : float
        Initial learning rate
    optimizer : torch.optim.Optimizer
        model optimizer
    max_grad_norm : float or None
        If not None, gradient clipping will be performed
    """
    def __init__(self, model, lr, optimizer, max_grad_norm=None):
        super(Optimizer, self).__init__()
        self.model = model
        self.lr = lr
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self._reset()

    def _reset(self):
        self.optimizer.zero_grad()

    def _clip_grad_norm(self):
        grad_norm = None
        if self.max_grad_norm is not None:
            grad_norm = clip_grad_norm_(self.model.parameters(),
                                        self.max_grad_norm)
        return grad_norm

    def backward_and_step(self, loss):
        """Backward and update model.

        Parameters
        ----------
        loss : torch.tensor consisting of a float only

        Returns
        -------
        grad_norm : float
            Gradient norm. If self.max_grad_norm is None, None will be returned.
        """
        self._reset()
        loss.backward()
        grad_norm = self._clip_grad_norm()
        self.optimizer.step()

        return grad_norm

    def decay_lr(self, decay_rate):
        """Decay learning rate.

        Parameters
        ----------
        decay_rate : float
            Multiply the current learning rate by the decay_rate
        """
        self.lr *= decay_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

class MultiProcessOptimizer(Optimizer):
    """Wrapper for optimization with multiprocess

    Parameters
    ----------
    n_processes : int
        Number of processes used
    model : nn.Module
        Model being trained
    lr : float
        Initial learning rate
    optimizer : torch.optim.Optimizer
        model optimizer
    max_grad_norm : float or None
        If not None, gradient clipping will be performed.
    """
    def __init__(self, n_processes, model, lr, optimizer, max_grad_norm=None):
        super(MultiProcessOptimizer, self).__init__(lr=lr, model=model, optimizer=optimizer,
                                                    max_grad_norm=max_grad_norm)
        self.n_processes = n_processes

    def _sync_gradient(self):
        """Average gradients across all subprocesses."""
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                if p.requires_grad and p.grad is not None:
                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                    p.grad.data /= self.n_processes

    def backward_and_step(self, loss):
        """Backward and update model.

        Parameters
        ----------
        loss : torch.tensor consisting of a float only

        Returns
        -------
        grad_norm : float
            Gradient norm. If self.max_grad_norm is None, None will be returned.
        """
        loss.backward()
        self._sync_gradient()
        grad_norm = self._clip_grad_norm()
        self.optimizer.step()
        self._reset()

        return grad_norm

def synchronize(num_gpus):
    """Synchronize all processes for multi-gpu training.

    Parameters
    ----------
    num_gpus : int
        Number of gpus used
    """
    if num_gpus > 1:
        dist.barrier()

def collate_center(data):
    """Collate multiple datapoints for reaction center prediction

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
    batch_mol_graphs : DGLGraph
        DGLGraph for a batch of molecular graphs.
    batch_complete_graphs : DGLGraph
        DGLGraph for a batch of complete graphs.
    batch_atom_pair_labels : float32 tensor of shape (V, 10)
        Labels of atom pairs in the batch of graphs.
    """
    reactions, graph_edits, mol_graphs, complete_graphs, \
    atom_pair_feats, atom_pair_labels = map(list, zip(*data))

    batch_mol_graphs = dgl.batch(mol_graphs)
    batch_mol_graphs.set_n_initializer(dgl.init.zero_initializer)
    batch_mol_graphs.set_e_initializer(dgl.init.zero_initializer)

    batch_complete_graphs = dgl.batch(complete_graphs)
    batch_complete_graphs.set_n_initializer(dgl.init.zero_initializer)
    batch_complete_graphs.set_e_initializer(dgl.init.zero_initializer)
    batch_complete_graphs.edata['feats'] = torch.cat(atom_pair_feats, dim=0)

    batch_atom_pair_labels = torch.cat(atom_pair_labels, dim=0)

    return reactions, graph_edits, batch_mol_graphs, \
           batch_complete_graphs, batch_atom_pair_labels

def reaction_center_prediction(device, model, mol_graphs, complete_graphs):
    """Perform a soft prediction on reaction center.

    Parameters
    ----------
    device : str
        Device to use for computation, e.g. 'cpu', 'cuda:0'
    model : nn.Module
        Model for prediction.
    mol_graphs : DGLGraph
        DGLGraph for a batch of molecular graphs
    complete_graphs : DGLGraph
        DGLGraph for a batch of complete graphs

    Returns
    -------
    scores : float32 tensor of shape (E_full, 5)
        Predicted scores for each pair of atoms to perform one of the following
        5 actions in reaction:

        * The bond between them gets broken
        * Forming a single bond
        * Forming a double bond
        * Forming a triple bond
        * Forming an aromatic bond
    biased_scores : float32 tensor of shape (E_full, 5)
        Comparing to scores, a bias is added if the pair is for a same atom.
    """
    node_feats = mol_graphs.ndata.pop('hv').to(device)
    edge_feats = mol_graphs.edata.pop('he').to(device)
    node_pair_feats = complete_graphs.edata.pop('feats').to(device)

    return model(mol_graphs, complete_graphs, node_feats, edge_feats, node_pair_feats)

def reaction_center_rough_eval(complete_graphs, preds, labels, num_correct):
    batch_size = complete_graphs.batch_size
    start = 0
    for i in range(batch_size):
        end = start + complete_graphs.batch_num_edges[i]
        preds_i = preds[start:end, :].flatten()
        labels_i = labels[start:end, :].flatten()
        for k in num_correct.keys():
            topk_values, topk_indices = torch.topk(preds_i, k)
            is_correct = labels_i[topk_indices].sum() == labels_i.sum().float().cpu().data.item()
            num_correct[k].append(is_correct)
        start = end

def reaction_center_rough_eval_on_a_loader(args, model, data_loader):
    """A rough evaluation of model performance in the middle of training.

    For final evaluation, we will eliminate some possibilities based on prior knowledge.

    Parameters
    ----------
    args : dict
        Configurations fot the experiment.
    model : nn.Module
        Model for reaction center prediction.
    data_loader : torch.utils.data.DataLoader
        Loader for fetching and batching data.

    Returns
    -------
    str
        Message for evluation result.
    """
    model.eval()
    num_correct = {k: [] for k in args['top_ks']}
    for batch_id, batch_data in enumerate(data_loader):
        batch_reactions, batch_graph_edits, batch_mol_graphs, \
        batch_complete_graphs, batch_atom_pair_labels = batch_data
        with torch.no_grad():
            pred, biased_pred = reaction_center_prediction(
                args['device'], model, batch_mol_graphs, batch_complete_graphs)
        reaction_center_rough_eval(
            batch_complete_graphs, biased_pred, batch_atom_pair_labels, num_correct)

    msg = '|'
    for k, correct_count in num_correct.items():
        msg += ' acc@{:d} {:.4f} |'.format(k, np.mean(correct_count))

    return msg

bond_change_to_id = {0.0: 0, 1:1, 2:2, 3:3, 1.5:4}
id_to_bond_change = {v: k for k, v in bond_change_to_id.items()}
num_change_types = len(bond_change_to_id)

def get_candidate_bonds(reaction, preds, num_nodes, max_k, easy, include_scores=False):
    """Get candidate bonds for a reaction.

    Parameters
    ----------
    reaction : str
        Reaction
    preds : float32 tensor of shape (E * 5)
        E for the number of edges in a complete graph and 5 for the number of possible
        bond changes.
    num_nodes : int
        Number of nodes in the graph.
    max_k : int
        Maximum number of atom pairs to be selected.
    easy : bool
        If True, reactants not contributing atoms to the product will be excluded in
        top-k atom pair selection, which will make the task easier.
    include_scores : bool
        Whether to include the scores for the atom pairs selected. Default to False.

    Returns
    -------
    list of 3-tuples or 4-tuples
        The first three elements in a tuple separately specify the first atom,
        the second atom and the type for bond change. If include_scores is True,
        the score for the prediction will be included as a fourth element.
    """
    # Decide which atom-pairs will be considered.
    reaction_atoms = []
    reaction_bonds = defaultdict(bool)
    reactants, _, product = reaction.split('>')
    product_mol = Chem.MolFromSmiles(product)
    product_atoms = set([atom.GetAtomMapNum() for atom in product_mol.GetAtoms()])

    for reactant in reactants.split('.'):
        reactant_mol = Chem.MolFromSmiles(reactant)
        reactant_atoms = [atom.GetAtomMapNum() for atom in reactant_mol.GetAtoms()]
        # In the hard mode, all reactant atoms will be included.
        # In the easy mode, only reactants contributing atoms to the product will be included.
        if (len(set(reactant_atoms) & product_atoms) > 0) or (not easy):
            reaction_atoms.extend(reactant_atoms)
            for bond in reactant_mol.GetBonds():
                end_atoms = sorted([bond.GetBeginAtom().GetAtomMapNum(),
                                    bond.GetEndAtom().GetAtomMapNum()])
                bond = tuple(end_atoms + [bond.GetBondTypeAsDouble()])
                # Bookkeep bonds already in reactants
                reaction_bonds[bond] = True

    candidate_bonds = []
    topk_values, topk_indices = torch.topk(preds, max_k)
    for j in range(max_k):
        preds_j = topk_indices[j].cpu().item()
        # A bond change can be either losing the bond or forming a
        # single, double, triple or aromatic bond
        change_id = preds_j % num_change_types
        change_type = id_to_bond_change[change_id]
        pair_id = preds_j // num_change_types
        # Atom map numbers
        atom1 = pair_id // num_nodes + 1
        atom2 = pair_id % num_nodes + 1
        # Avoid duplicates and an atom cannot form a bond with itself
        if atom1 >= atom2:
            continue
        if atom1 not in reaction_atoms:
            continue
        if atom2 not in reaction_atoms:
            continue
        candidate = (int(atom1), int(atom2), float(change_type))
        if reaction_bonds[candidate]:
            continue
        if include_scores:
            candidate += (float(topk_values[j].cpu().item()),)
        candidate_bonds.append(candidate)

    return candidate_bonds

def reaction_center_eval(complete_graphs, preds, reactions,
                         graph_edits, num_correct, max_k, easy):
    """Evaluate top-k accuracies for reaction center prediction.

    Parameters
    ----------
    complete_graphs : DGLGraph
        DGLGraph for a batch of complete graphs
    preds : float32 tensor of shape (E_full, 5)
        Soft predictions for reaction center, E_full being the number of possible
        atom-pairs and 5 being the number of possible bond changes
    reactions : list of str
        List of reactions.
    graph_edits : list of str
        List of graph edits in the reactions.
    num_correct : dict
        Counting the number of datapoints for meeting top-k accuracies.
    max_k : int
        Maximum number of atom pairs to be selected. This is intended to be larger
        than max(num_correct.keys()) as we will filter out many atom pairs due to
        considerations such as avoiding duplicates.
    easy : bool
        If True, reactants not contributing atoms to the product will be excluded in
        top-k atom pair selection, which will make the task easier.
    """
    # 0 for losing the bond
    # 1, 2, 3, 1.5 separately for forming a single, double, triple or aromatic bond.
    batch_size = complete_graphs.batch_size
    start = 0
    for i in range(batch_size):
        end = start + complete_graphs.batch_num_edges[i]
        candidate_bonds = get_candidate_bonds(
            reactions[i], preds[start:end, :].flatten(),
            complete_graphs.batch_num_nodes[i], max_k, easy)

        gold_bonds = []
        gold_edits = graph_edits[i]
        for edit in gold_edits.split(';'):
            atom1, atom2, change_type = edit.split('-')
            atom1, atom2 = int(atom1), int(atom2)
            gold_bonds.append((min(atom1, atom2), max(atom1, atom2), float(change_type)))

        for k in num_correct.keys():
            if set(gold_bonds) <= set(candidate_bonds[:k]):
                num_correct[k] += 1
        start = end

def reaction_center_final_eval(args, model, data_loader, easy):
    """Final evaluation of model performance.

    args : dict
        Configurations fot the experiment.
    model : nn.Module
        Model for reaction center prediction.
    data_loader : torch.utils.data.DataLoader
        Loader for fetching and batching data.
    easy : bool
        If True, reactants not contributing atoms to the product will be excluded in
        top-k atom pair selection, which will make the task easier.

    Returns
    -------
    msg : str
        Summary of the top-k evaluation.
    """
    model.eval()
    num_correct = {k: 0 for k in args['top_ks']}
    for batch_id, batch_data in enumerate(data_loader):
        batch_reactions, batch_graph_edits, batch_mol_graphs, \
        batch_complete_graphs, batch_atom_pair_labels = batch_data
        with torch.no_grad():
            pred, biased_pred = reaction_center_prediction(
                args['device'], model, batch_mol_graphs, batch_complete_graphs)
        reaction_center_eval(batch_complete_graphs, biased_pred, batch_reactions,
             batch_graph_edits, num_correct, args['max_k'], easy)

    msg = '|'
    for k, correct_count in num_correct.items():
        msg += ' acc@{:d} {:.4f} |'.format(k, correct_count / len(data_loader.dataset))

    return msg

def output_candidate_bonds_for_a_reaction(info, max_k):
    """Prepare top-k atom pairs for each reaction as candidate bonds

    Parameters
    ----------
    info : 3-tuple for a reaction
        Consists of the reaction, the scores for atom-pairs in reactants
        and the number of nodes in reactants.

    Returns
    -------
    candidate_string : str
        String representing candidate bonds for a reaction. Each candidate
        bond is of format 'atom1 atom2 change_type score'.
    """
    reaction, preds, num_nodes = info
    # Note that we use the easy mode by default, which is also the
    # setting in the paper.
    candidate_bonds = get_candidate_bonds(reaction, preds, num_nodes, max_k,
                                          easy=True, include_scores=True)
    candidate_string = ''
    for candidate in candidate_bonds:
        # A 4-tuple consisting of the atom mapping number of atom 1,
        # atom 2, the bond change type and the score
        candidate_string += '{} {} {:.1f} {:.3f};'.format(
            candidate[0], candidate[1], candidate[2], candidate[3])
    candidate_string += '\n'

    return candidate_string

def prepare_reaction_center(args, reaction_center_config):
    """Use a trained model for reaction center prediction to prepare candidate bonds.

    Parameters
    ----------
    args : dict
        Configuration for the experiment.
    reaction_center_config : dict
        Configuration for the experiment on reaction center prediction.

    Returns
    -------
    path_to_candidate_bonds : dict
        Mapping 'train', 'val', 'test' to the corresponding files for candidate bonds.
    """
    if args['center_model_path'] is None:
        reaction_center_model = load_pretrained('wln_center_uspto').to(args['device'])
    else:
        reaction_center_model = WLNReactionCenter(
            node_in_feats=reaction_center_config['node_in_feats'],
            edge_in_feats=reaction_center_config['edge_in_feats'],
            node_pair_in_feats=reaction_center_config['node_pair_in_feats'],
            node_out_feats=reaction_center_config['node_out_feats'],
            n_layers=reaction_center_config['n_layers'],
            n_tasks=reaction_center_config['n_tasks'])
        reaction_center_model.load_state_dict(
            torch.load(args['center_model_path'])['model_state_dict'])
        reaction_center_model = reaction_center_model.to(args['device'])
    reaction_center_model.eval()

    path_to_candidate_bonds = dict()
    for subset in ['train', 'val', 'test']:
        if '{}_path'.format(subset) not in args:
            continue

        path_to_candidate_bonds[subset] = args['result_path'] + \
                                          '/{}_candidate_bonds.txt'.format(subset)
        if os.path.isfile(path_to_candidate_bonds[subset]):
            continue

        print('Processing subset {}...'.format(subset))
        print('Stage 1/3: Loading dataset...')
        if args['{}_path'.format(subset)] is None:
            dataset = USPTOCenter(subset, num_processes=args['num_processes'])
        else:
            dataset = WLNCenterDataset(raw_file_path=args['{}_path'.format(subset)],
                                       mol_graph_path='{}.bin'.format(subset),
                                       num_processes=args['num_processes'])

        dataloader = DataLoader(dataset, batch_size=args['reaction_center_batch_size'],
                                collate_fn=collate, shuffle=False)

        print('Stage 2/3: Performing model prediction...')
        output_strings = []
        for batch_id, batch_data in enumerate(dataloader):
            print('Computing candidate bonds for batch {:d}/{:d}'.format(
                batch_id + 1, len(dataloader)))
            batch_reactions, batch_graph_edits, batch_mol_graphs, \
            batch_complete_graphs, batch_atom_pair_labels = batch_data
            with torch.no_grad():
                pred, biased_pred = reaction_center_prediction(
                    args['device'], reaction_center_model,
                    batch_mol_graphs, batch_complete_graphs)
            batch_size = len(batch_reactions)
            start = 0
            for i in range(batch_size):
                end = start + batch_complete_graphs.batch_num_edges[i]
                output_strings.append(output_candidate_bonds_for_a_reaction(
                    (batch_reactions[i], biased_pred[start:end, :].flatten(),
                    batch_complete_graphs.batch_num_nodes[i]), reaction_center_config['max_k']
                ))
                start = end

        print('Stage 3/3: Output candidate bonds...')
        with open(path_to_candidate_bonds[subset], 'w') as f:
            for candidate_string in output_strings:
                f.write(candidate_string)

        del dataset
        del dataloader

    return path_to_candidate_bonds

def collate_rank(data):
    """Collate multiple datapoints for candidate product ranking

    Parameters
    ----------
    data : list of 4-tuples or 5-tuples
        Each tuple is for a single datapoint, consisting of combos of bond changes for candidate
        products, DGLGraphs for reactants and candidate products, node features shared across
        all DGLGraphs, scores for candidate products by the model for reaction center prediction,
        and labels for candidate products.

    Returns
    -------
    candidate_combos
    """
    assert len(data) == 1, 'This collate function only works with batch size 1.'
    data = data[0]
    if len(data) == 4:
        candidate_combos, graphs, node_feats, combo_scores = data
    else:
        candidate_combos, graphs, node_feats, combo_scores, labels = data
    bg = dgl.batch(graphs)
    edge_feats = bg.edata.pop('he')
    old_feats_shape = node_feats.shape
    node_feats = node_feats.reshape((1,) + old_feats_shape)
    node_feats = node_feats.expand((bg.batch_size,) + old_feats_shape)

    if len(data) == 4:
        return candidate_combos, bg, node_feats, edge_feats, combo_scores
    else:
        return candidate_combos, bg, node_feats, edge_feats, combo_scores, labels
