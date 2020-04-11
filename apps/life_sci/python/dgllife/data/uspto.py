"""USPTO for reaction prediction"""
import numpy as np
import os
import random
import torch

from collections import defaultdict
from dgl.data.utils import get_download_dir, download, _get_dgl_url, extract_archive, \
    save_graphs, load_graphs
from functools import partial
from itertools import combinations
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolops
from tqdm import tqdm

from ..utils.featurizers import BaseAtomFeaturizer, ConcatFeaturizer, atom_type_one_hot, \
    atom_degree_one_hot, atom_explicit_valence_one_hot, atom_implicit_valence_one_hot, \
    atom_is_aromatic, atom_formal_charge_one_hot, BaseBondFeaturizer, bond_type_one_hot, \
    bond_is_conjugated, bond_is_in_ring
from ..utils.mol_to_graph import mol_to_bigraph, mol_to_complete_graph

__all__ = ['WLNCenterDataset',
           'USPTOCenter',
           'WLNRankDataset',
           'USPTORank']

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Atom types distinguished in featurization
atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
              'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
              'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
              'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi',
              'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs']

default_node_featurizer_center = BaseAtomFeaturizer({
    'hv': ConcatFeaturizer(
        [partial(atom_type_one_hot,
                 allowable_set=atom_types, encode_unknown=True),
         partial(atom_degree_one_hot,
                 allowable_set=list(range(5)), encode_unknown=True),
         partial(atom_explicit_valence_one_hot,
                 allowable_set=list(range(1, 6)), encode_unknown=True),
         partial(atom_implicit_valence_one_hot,
                 allowable_set=list(range(5)), encode_unknown=True),
         atom_is_aromatic]
    )
})

default_node_featurizer_rank = BaseAtomFeaturizer({
    'hv': ConcatFeaturizer(
        [partial(atom_type_one_hot,
                 allowable_set=atom_types, encode_unknown=True),
         partial(atom_formal_charge_one_hot,
                 allowable_set=[-3, -2, -1, 0, 1, 2], encode_unknown=True),
         partial(atom_degree_one_hot,
                 allowable_set=list(range(5)), encode_unknown=True),
         partial(atom_explicit_valence_one_hot,
                 allowable_set=list(range(1, 6)), encode_unknown=True),
         partial(atom_implicit_valence_one_hot,
                 allowable_set=list(range(5)), encode_unknown=True),
         atom_is_aromatic]
    )
})

default_edge_featurizer_center = BaseBondFeaturizer({
    'he': ConcatFeaturizer([
        bond_type_one_hot, bond_is_conjugated, bond_is_in_ring]
    )
})

default_edge_featurizer_rank = BaseBondFeaturizer({
    'he': ConcatFeaturizer([
        bond_type_one_hot, bond_is_in_ring]
    )
})

def default_atom_pair_featurizer(reactants):
    """Featurize each pair of atoms, which will be used in updating
    the edata of a complete DGLGraph.

    The features include the bond type between the atoms (if any) and whether
    they belong to the same molecule. It is used in the global attention mechanism.

    Parameters
    ----------
    reactants : str
        SMILES for reactants
    data_field : str
        Key for storing the features in DGLGraph.edata. Default to 'atom_pair'

    Returns
    -------
    float32 tensor of shape (V^2, 10)
        features for each pair of atoms.
    """
    # Decide the reactant membership for each atom
    atom_to_reactant = dict()
    reactant_list = reactants.split('.')
    for id, s in enumerate(reactant_list):
        mol = Chem.MolFromSmiles(s)
        for atom in mol.GetAtoms():
            atom_to_reactant[atom.GetIntProp('molAtomMapNumber') - 1] = id

    # Construct mapping from atom pair to RDKit bond object
    all_reactant_mol = Chem.MolFromSmiles(reactants)
    atom_pair_to_bond = dict()
    for bond in all_reactant_mol.GetBonds():
        atom1 = bond.GetBeginAtom().GetIntProp('molAtomMapNumber') - 1
        atom2 = bond.GetEndAtom().GetIntProp('molAtomMapNumber') - 1
        atom_pair_to_bond[(atom1, atom2)] = bond
        atom_pair_to_bond[(atom2, atom1)] = bond

    def _featurize_a_bond(bond):
        return bond_type_one_hot(bond) + bond_is_conjugated(bond) + bond_is_in_ring(bond)

    features = []
    num_atoms = all_reactant_mol.GetNumAtoms()
    for i in range(num_atoms):
        for j in range(num_atoms):
            pair_feature = np.zeros(10)
            if i == j:
                features.append(pair_feature)
                continue

            bond = atom_pair_to_bond.get((i, j), None)
            if bond is not None:
                pair_feature[1:7] = _featurize_a_bond(bond)
            else:
                pair_feature[0] = 1.
            pair_feature[-4] = 1. if atom_to_reactant[i] != atom_to_reactant[j] else 0.
            pair_feature[-3] = 1. if atom_to_reactant[i] == atom_to_reactant[j] else 0.
            pair_feature[-2] = 1. if len(reactant_list) == 1 else 0.
            pair_feature[-1] = 1. if len(reactant_list) > 1 else 0.
            features.append(pair_feature)

    return torch.from_numpy(np.stack(features, axis=0).astype(np.float32))

def get_pair_label(reactants_mol, graph_edits):
    """Construct labels for each pair of atoms in reaction center prediction

    Parameters
    ----------
    reactants_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for all reactants in a reaction
    graph_edits : str
        Specifying which pairs of atoms loss a bond or form a particular bond in the reaction

    Returns
    -------
    float32 tensor of shape (V^2, 5)
        Labels constructed. V for the number of atoms in the reactants.
    """
    # 0 for losing the bond
    # 1, 2, 3, 1.5 separately for forming a single, double, triple or aromatic bond.
    bond_change_to_id = {0.0: 0, 1:1, 2:2, 3:3, 1.5:4}
    pair_to_changes = defaultdict(list)
    for edit in graph_edits.split(';'):
        a1, a2, change = edit.split('-')
        atom1 = int(a1) - 1
        atom2 = int(a2) - 1
        change = bond_change_to_id[float(change)]
        pair_to_changes[(atom1, atom2)].append(change)
        pair_to_changes[(atom2, atom1)].append(change)

    num_atoms = reactants_mol.GetNumAtoms()
    labels = torch.zeros((num_atoms, num_atoms, 5))
    for pair in pair_to_changes.keys():
        i, j = pair
        labels[i, j, pair_to_changes[(j, i)]] = 1.

    return labels.reshape(-1, 5)

def get_bond_changes(reaction):
    """Get the bond changes in a reaction.

    Parameters
    ----------
    reaction : str
        SMILES for a reaction, e.g. [CH3:14][NH2:15].[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7]
        (=[O:8])[OH:9])[cH:10][cH:11][c:12]1[Cl:13].[OH2:16]>>[N+:1](=[O:2])([O-:3])[c:4]1[cH:5]
        [c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[NH:15][CH3:14]. It consists of reactants,
        products and the atom mapping.

    Returns
    -------
    bond_changes : set of 3-tuples
        Each tuple consists of (atom1, atom2, change type)
        There are 5 possible values for change type. 0 for losing the bond, and 1, 2, 3, 1.5
        separately for forming a single, double, triple or aromatic bond.
    """
    reactants = Chem.MolFromSmiles(reaction.split('>')[0])
    products  = Chem.MolFromSmiles(reaction.split('>')[2])

    conserved_maps = [
        a.GetProp('molAtomMapNumber')
        for a in products.GetAtoms() if a.HasProp('molAtomMapNumber')]
    bond_changes = set() # keep track of bond changes

    # Look at changed bonds
    bonds_prev = {}
    for bond in reactants.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetProp('molAtomMapNumber'),
             bond.GetEndAtom().GetProp('molAtomMapNumber')])
        if (nums[0] not in conserved_maps) and (nums[1] not in conserved_maps):
            continue
        bonds_prev['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()
    bonds_new = {}
    for bond in products.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetProp('molAtomMapNumber'),
             bond.GetEndAtom().GetProp('molAtomMapNumber')])
        bonds_new['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()

    for bond in bonds_prev:
        if bond not in bonds_new:
            # lost bond
            bond_changes.add((bond.split('~')[0], bond.split('~')[1], 0.0))
        else:
            if bonds_prev[bond] != bonds_new[bond]:
                # changed bond
                bond_changes.add((bond.split('~')[0], bond.split('~')[1], bonds_new[bond]))
    for bond in bonds_new:
        if bond not in bonds_prev:
            # new bond
            bond_changes.add((bond.split('~')[0], bond.split('~')[1], bonds_new[bond]))

    return bond_changes

def process_file(path):
    """Pre-process a file of reactions for working with WLN.

    Parameters
    ----------
    path : str
        Path to the file of reactions
    """
    with open(path, 'r') as input_file, open(path + '.proc', 'w') as output_file:
        for line in tqdm(input_file):
            reaction = line.strip()
            bond_changes = get_bond_changes(reaction)
            output_file.write('{} {}\n'.format(
                reaction,
                ';'.join(['{}-{}-{}'.format(x[0], x[1], x[2]) for x in bond_changes])))
    print('Finished processing {}'.format(path))

class WLNCenterDataset(object):
    """Dataset for reaction center prediction with WLN

    Parameters
    ----------
    raw_file_path : str
        Path to the raw reaction file, where each line is the SMILES for a reaction.
        We will check if raw_file_path + '.proc' exists, where each line has the reaction
        SMILES and the corresponding graph edits. If not, we will preprocess
        the raw reaction file.
    mol_graph_path : str
        Path to save/load DGLGraphs for molecules.
    mol_to_graph: callable, str -> DGLGraph
        A function turning RDKit molecule instances into DGLGraphs.
        Default to :func:`dgllife.utils.mol_to_bigraph`.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. By default, we consider descriptors including atom type,
        atom degree, atom explicit valence, atom implicit valence, aromaticity.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. By default, we consider descriptors including bond type,
        whether bond is conjugated and whether bond is in ring.
    atom_pair_featurizer : callable, str -> dict
        Featurization for each pair of atoms in multiple reactants. The result will be
        used to update edata in the complete DGLGraphs. By default, the features include
        the bond type between the atoms (if any) and whether they belong to the same molecule.
    load : bool
        Whether to load the previously pre-processed dataset or pre-process from scratch.
        ``load`` should be False when we want to try different graph construction and
        featurization methods and need to preprocess from scratch. Default to True.
    log_every : int
        Print a progress update every time ``log_every`` reactions are pre-processed.
        Default to 10000.
    """
    def __init__(self,
                 raw_file_path,
                 mol_graph_path,
                 mol_to_graph=mol_to_bigraph,
                 node_featurizer=default_node_featurizer_center,
                 edge_featurizer=default_edge_featurizer_center,
                 atom_pair_featurizer=default_atom_pair_featurizer,
                 load=True,
                 log_every=10000):
        super(WLNCenterDataset, self).__init__()

        self._atom_pair_featurizer = atom_pair_featurizer
        self.atom_pair_features = []
        self.atom_pair_labels = []
        # Map number of nodes to a corresponding complete graph
        self.complete_graphs = dict()

        path_to_reaction_file = raw_file_path + '.proc'
        if not os.path.isfile(path_to_reaction_file):
            # Pre-process graph edits information
            process_file(raw_file_path)

        full_mols, full_reactions, full_graph_edits = \
            self.load_reaction_data(path_to_reaction_file, log_every)
        if load and os.path.isfile(mol_graph_path):
            print('Loading previously saved graphs...')
            self.reactant_mol_graphs, _ = load_graphs(mol_graph_path)
        else:
            print('Constructing graphs from scratch...')
            self.reactant_mol_graphs = []
            for i in range(len(full_mols)):
                if i % log_every == 0:
                    print('Processing reaction {:d}/{:d}'.format(i + 1, len(full_mols)))
                mol = full_mols[i]
                reactant_mol_graph = mol_to_graph(mol, node_featurizer=node_featurizer,
                                                  edge_featurizer=edge_featurizer,
                                                  canonical_atom_order=False)
                self.reactant_mol_graphs.append(reactant_mol_graph)

            save_graphs(mol_graph_path, self.reactant_mol_graphs)

        self.mols = full_mols
        self.reactions = full_reactions
        self.graph_edits = full_graph_edits
        self.atom_pair_features.extend([None for _ in range(len(self.mols))])
        self.atom_pair_labels.extend([None for _ in range(len(self.mols))])

    def load_reaction_data(self, file_path, log_every):
        """Load reaction data from the raw file.

        Parameters
        ----------
        file_path : str
            Path to read the file.
        log_every : int
            Print a progress update every time ``log_every`` reactions are pre-processed.

        Returns
        -------
        all_mols : list of rdkit.Chem.rdchem.Mol
            RDKit molecule instances
        all_reactions : list of str
            Reactions
        all_graph_edits : list of str
            Graph edits in the reactions.
        """
        all_mols = []
        all_reactions = []
        all_graph_edits = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i % log_every == 0:
                    print('Processing line {:d}'.format(i))
                # Each line represents a reaction and the corresponding graph edits
                #
                # reaction example:
                # [CH3:14][OH:15].[NH2:12][NH2:13].[OH2:11].[n:1]1[n:2][cH:3][c:4]
                # ([C:7]([O:9][CH3:8])=[O:10])[cH:5][cH:6]1>>[n:1]1[n:2][cH:3][c:4]
                # ([C:7](=[O:9])[NH:12][NH2:13])[cH:5][cH:6]1
                # The reactants are on the left-hand-side of the reaction and the product
                # is on the right-hand-side of the reaction. The numbers represent atom mapping.
                #
                # graph_edits example:
                # 23-33-1.0;23-25-0.0
                # For a triplet a-b-c, a and b are the atoms that form or loss the bond.
                # c specifies the particular change, 0.0 for losing a bond, 1.0, 2.0, 3.0 and
                # 1.5 separately for forming a single, double, triple or aromatic bond.
                reaction, graph_edits = line.strip("\r\n ").split()
                reactants = reaction.split('>')[0]
                mol = Chem.MolFromSmiles(reactants)
                if mol is None:
                    continue

                # Reorder atoms according to the order specified in the atom map
                atom_map_order = [-1 for _ in range(mol.GetNumAtoms())]
                for i in range(mol.GetNumAtoms()):
                    atom = mol.GetAtomWithIdx(i)
                    atom_map_order[atom.GetIntProp('molAtomMapNumber') - 1] = i
                mol = rdmolops.RenumberAtoms(mol, atom_map_order)
                all_mols.append(mol)
                all_reactions.append(reaction)
                all_graph_edits.append(graph_edits)

        return all_mols, all_reactions, all_graph_edits

    def __len__(self):
        """Get the size for the dataset.

        Returns
        -------
        int
            Number of reactions in the dataset.
        """
        return len(self.mols)

    def __getitem__(self, item):
        """Get the i-th datapoint.

        Returns
        -------
        str
            Reaction
        str
            Graph edits for the reaction
        rdkit.Chem.rdchem.Mol
            RDKit molecule instance
        DGLGraph
            DGLGraph for the ith molecular graph
        DGLGraph
            Complete DGLGraph, which will be needed for predicting
            scores between each pair of atoms
        float32 tensor of shape (V^2, 10)
            Features for each pair of atoms.
        float32 tensor of shape (V^2, 5)
            Labels for reaction center prediction.
            V for the number of atoms in the reactants.
        """
        mol = self.mols[item]
        num_atoms = mol.GetNumAtoms()

        if num_atoms not in self.complete_graphs:
            self.complete_graphs[num_atoms] = mol_to_complete_graph(
                mol, add_self_loop=True, canonical_atom_order=True)

        if self.atom_pair_features[item] is None:
            reactants = self.reactions[item].split('>')[0]
            self.atom_pair_features[item] = self._atom_pair_featurizer(reactants)

        if self.atom_pair_labels[item] is None:
            self.atom_pair_labels[item] = get_pair_label(mol, self.graph_edits[item])

        return self.reactions[item], self.graph_edits[item], mol, \
               self.reactant_mol_graphs[item], \
               self.complete_graphs[num_atoms], \
               self.atom_pair_features[item], \
               self.atom_pair_labels[item]

class USPTOCenter(WLNCenterDataset):
    """USPTO dataset for reaction center prediction.

    The dataset contains reactions from patents granted by United States Patent
    and Trademark Office (USPTO), collected by Lowe [1]. Jin et al. removes duplicates
    and erroneous reactions, obtaining a set of 480K reactions. They divide it
    into 400K, 40K, and 40K for training, validation and test.

    References:

        * [1] Patent reaction extraction
        * [2] Predicting Organic Reaction Outcomes with Weisfeiler-Lehman Network

    Parameters
    ----------
    subset : str
        Whether to use the training/validation/test set as in Jin et al.

        * 'train' for the training set
        * 'val' for the validation set
        * 'test' for the test set
    mol_to_graph: callable, str -> DGLGraph
        A function turning RDKit molecule instances into DGLGraphs.
        Default to :func:`dgllife.utils.mol_to_bigraph`.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. By default, we consider descriptors including atom type,
        atom degree, atom explicit valence, atom implicit valence, aromaticity.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. By default, we consider descriptors including bond type,
        whether bond is conjugated and whether bond is in ring.
    atom_pair_featurizer : callable, str -> dict
        Featurization for each pair of atoms in multiple reactants. The result will be
        used to update edata in the complete DGLGraphs. By default, the features include
        the bond type between the atoms (if any) and whether they belong to the same molecule.
    load : bool
        Whether to load the previously pre-processed dataset or pre-process from scratch.
        ``load`` should be False when we want to try different graph construction and
        featurization methods and need to preprocess from scratch. Default to True.
    """
    def __init__(self,
                 subset,
                 mol_to_graph=mol_to_bigraph,
                 node_featurizer=default_node_featurizer_center,
                 edge_featurizer=default_edge_featurizer_center,
                 atom_pair_featurizer=default_atom_pair_featurizer,
                 load=True):
        assert subset in ['train', 'val', 'test'], \
            'Expect subset to be "train" or "val" or "test", got {}'.format(subset)
        print('Preparing {} subset of USPTO for reaction center prediction.'.format(subset))
        self._subset = subset
        if subset == 'val':
            subset = 'valid'

        self._url = 'dataset/uspto.zip'
        data_path = get_download_dir() + '/uspto.zip'
        extracted_data_path = get_download_dir() + '/uspto'
        download(_get_dgl_url(self._url), path=data_path)
        extract_archive(data_path, extracted_data_path)

        super(USPTOCenter, self).__init__(
            raw_file_path=extracted_data_path + '/{}.txt'.format(subset),
            mol_graph_path=extracted_data_path + '/{}_mol_graphs.bin'.format(subset),
            mol_to_graph=mol_to_graph,
            node_featurizer=node_featurizer,
            edge_featurizer=edge_featurizer,
            atom_pair_featurizer=atom_pair_featurizer,
            load=load)

    @property
    def subset(self):
        """Get the subset used for USPTOCenter

        Returns
        -------
        str

            * 'full' for the complete dataset
            * 'train' for the training set
            * 'val' for the validation set
            * 'test' for the test set
        """
        return self._subset

class WLNRankDataset(object):
    """Dataset for ranking candidate products with WLN

    Parameters
    ----------
    raw_file_path : str
        Path to the raw reaction file, where each line is the SMILES for a reaction and
        the corresponding graph edits.
    candidate_bond_path : str
        Path to the candidate bond changes for product enumeration, where each line is
        candidate bond changes for a reaction by a WLN for reaction center prediction.
    mol_graph_path : str
        Path to save/load DGLGraphs for molecules.
    mol_to_graph: callable, str -> DGLGraph
        A function turning RDKit molecule instances into DGLGraphs.
        Default to :func:`dgllife.utils.mol_to_bigraph`.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. By default, we consider descriptors including atom type,
        atom formal charge, atom degree, atom explicit valence, atom implicit valence,
        aromaticity.
    size_cutoff : int
        By calling ``.ignore_large(True)``, we can optionally ignore reactions whose reactants
        contain more than ``size_cutoff`` atoms. Default to 100.
    max_num_changes_per_reaction : int
        Maximum number of bond changes per reaction. Default to 5.
    num_candidate_bond_changes : int
        Number of candidate bond changes to consider for each ground truth reaction.
        Default to 16.
    max_num_change_combos_per_reaction : int
        Number of bond change combos to consider for each reaction. Default to 150.
    train_mode : bool
        Whether the dataset is to be used for training. Default to True.
    log_every : int
        Print a progress update every time ``log_every`` reactions are pre-processed.
        Default to 10000.
    load : bool
        Whether to load the previously pre-processed dataset or pre-process from scratch.
        ``load`` should be False when we want to try different graph construction and
        featurization methods and need to preprocess from scratch. Default to True.
    """
    def __init__(self,
                 raw_file_path,
                 candidate_bond_path,
                 mol_graph_path,
                 mol_to_graph=mol_to_bigraph,
                 node_featurizer=default_node_featurizer_rank,
                 size_cutoff=100,
                 max_num_changes_per_reaction=5,
                 num_candidate_bond_changes=16,
                 max_num_change_combos_per_reaction=150,
                 train_mode=True,
                 log_every=10000,
                 load=True):
        super(WLNRankDataset, self).__init__()

        self.ignore_large_samples = False
        self.size_cutoff = size_cutoff
        self.train_mode = train_mode

        self.reactant_mols, self.product_mols, self.reactions, self.graph_edits, \
        self.real_bond_changes, self.ids_for_small_samples = \
            self.load_reaction_data(raw_file_path, log_every)
        self.candidate_bond_changes = self.load_candidate_bond_changes(
            candidate_bond_path, log_every)
        self.valid_candidate_combos = self.pre_process(
            num_candidate_bond_changes, max_num_changes_per_reaction,
            max_num_change_combos_per_reaction)

        self.reactant_mol_graphs = []
        if load and os.path.isfile(mol_graph_path):
            print('Loading previously saved graphs...')
            reactant_mol_graphs_, _ = load_graphs(mol_graph_path)
            for graph in reactant_mol_graphs_:
                # Re-use graphs saved for reaction center prediction
                nkeys = list(graph.ndata.keys())
                for key in nkeys:
                    graph.ndata.pop(key)
                ekeys = list(graph.edata.keys())
                for key in ekeys:
                    graph.edata.pop(key)
                self.reactant_mol_graphs.append(graph)
        else:
            print('Constructing graphs from scratch...')
            for i in range(len(self.reactant_mols)):
                if i % log_every == 0:
                    print('Processing reaction {:d}/{:d}'.format(i + 1, len(self.reactant_mols)))
                mol = self.reactant_mols[i]
                graph =

    def load_reaction_data(self, file_path, log_every):
        """Load reaction data from the raw file.

        Parameters
        ----------
        file_path : str
            Path to read the file.
        log_every : int
            Print a progress update every time ``log_every`` reactions are pre-processed.

        Returns
        -------
        all_reactant_mols : list of rdkit.Chem.rdchem.Mol
            RDKit molecule instances for reactants.
        all_product_mols : list of rdkit.Chem.rdchem.Mol
            RDKit molecule instances for products if the dataset is for training and
            None otherwise.
        all_reactions : list of str
            Reactions
        all_graph_edits : list of str
            Graph edits in the reactions.
        all_real_bond_changes : list of list
            ``all_real_bond_changes[i]`` gives a list of tuples, which are ground
            truth bond changes for a reaction.
        ids_for_small_samples : list of int
            Indices for reactions whose reactants do not contain too many atoms
        """
        all_reactant_mols = []
        if self.train_mode:
            all_product_mols = []
        else:
            all_product_mols = None
        all_reactions = []
        all_graph_edits = []
        all_real_bond_changes = []
        ids_for_small_samples = []
        curr_id = 0
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i % log_every == 0:
                    print('Processing line {:d}'.format(i))
                # Each line represents a reaction and the corresponding graph edits
                #
                # reaction example:
                # [CH3:14][OH:15].[NH2:12][NH2:13].[OH2:11].[n:1]1[n:2][cH:3][c:4]
                # ([C:7]([O:9][CH3:8])=[O:10])[cH:5][cH:6]1>>[n:1]1[n:2][cH:3][c:4]
                # ([C:7](=[O:9])[NH:12][NH2:13])[cH:5][cH:6]1
                # The reactants are on the left-hand-side of the reaction and the product
                # is on the right-hand-side of the reaction. The numbers represent atom mapping.
                #
                # graph_edits example:
                # 23-33-1.0;23-25-0.0
                # For a triplet a-b-c, a and b are the atoms that form or loss the bond.
                # c specifies the particular change, 0.0 for losing a bond, 1.0, 2.0, 3.0 and
                # 1.5 separately for forming a single, double, triple or aromatic bond.
                reaction, graph_edits = line.strip("\r\n ").split()
                reactants, _, product = reaction.split('>')
                reactants_mol = Chem.MolFromSmiles(reactants)
                if reactants_mol is None:
                    continue
                if self.train_mode:
                    product_mol = Chem.MolFromSmiles(product)
                    if product_mol is None:
                        continue
                    atom_map_order = [-1 for _ in range(product_mol.GetNumAtoms())]
                    for i in range(product_mol.GetNumAtoms()):
                        atom = product_mol.GetAtomWithIdx(i)
                        atom_map_order[atom.GetIntProp('molAtomMapNumber') - 1] = i
                    product_mol = rdmolops.RenumberAtoms(product_mol, atom_map_order)
                    all_product_mols.append(product_mol)
                if reactants_mol.GetNumAtoms() <= self.size_cutoff:
                    ids_for_small_samples.append(curr_id)

                # Reorder atoms according to the order specified in the atom map
                atom_map_order = [-1 for _ in range(reactants_mol.GetNumAtoms())]
                for i in range(reactants_mol.GetNumAtoms()):
                    atom = reactants_mol.GetAtomWithIdx(i)
                    atom_map_order[atom.GetIntProp('molAtomMapNumber') - 1] = i
                reactants_mol = rdmolops.RenumberAtoms(reactants_mol, atom_map_order)
                all_reactant_mols.append(reactants_mol)
                all_reactions.append(reaction)
                all_graph_edits.append(graph_edits)

                reaction_real_bond_changes = []
                for changed_bond in graph_edits.split(';'):
                    atom1, atom2, change_type = changed_bond.split('-')
                    atom1, atom2 = int(atom1) - 1, int(atom2) - 1
                    reaction_real_bond_changes.append(
                        (min(atom1, atom2), max(atom1, atom2), float(change_type)))
                all_real_bond_changes.append(reaction_real_bond_changes)

                curr_id += 1

        return all_reactant_mols, all_product_mols, all_reactions, all_graph_edits, \
               all_real_bond_changes, ids_for_small_samples

    def load_candidate_bond_changes(self, file_path, log_every):
        """Load candidate bond changes predicted by a WLN for reaction center prediction.

        Parameters
        ----------
        file_path : str
            Path to a file of candidate bond changes for each reaction.
        log_every : int
            Print a progress update every time ``log_every`` reactions are pre-processed.

        Returns
        -------
        all_candidate_bond_changes : list of list
            ``all_candidate_bond_changes[i]`` gives a list of tuples, which are candidate
            bond changes for a reaction.
        """
        all_candidate_bond_changes = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i % log_every == 0:
                    print('Processing line {:d}'.format(i))
                reaction_candidate_bond_changes = []
                elements = line.strip().split(';')[:-1]
                for candidate in elements:
                    atom1, atom2, change_type, score = candidate.split('-')
                    atom1, atom2 = int(atom1) - 1, int(atom2) - 1
                    reaction_candidate_bond_changes.append((
                        min(atom1, atom2), max(atom1, atom2), float(change_type), float(score)))
                all_candidate_bond_changes.append(reaction_candidate_bond_changes)

        return all_candidate_bond_changes

    def bookkeep_reactant(self, mol, candidate_pairs):
        """Bookkeep reaction-related information of reactants.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance for reactants.
        candidate_pairs : list of 2-tuples
            Pairs of atoms that ranked high by a model for reaction center prediction.
            By assumption, the two atoms are different and the first atom has a smaller
            index than the second.

        Returns
        -------
        info : dict
            Reaction-related information of reactants
        """
        num_atoms = mol.GetNumAtoms()
        info = {
            # free valence of atoms
            'free_val': [0 for _ in range(num_atoms)],
            # Whether it is a carbon atom
            'is_c': [False for _ in range(num_atoms)],
            # Whether it is a carbon atom connected to a nitrogen atom in pyridine
            'is_c2_of_pyridine': [False for _ in range(num_atoms)],
            # Whether it is a phosphorous atom
            'is_p': [False for _ in range(num_atoms)],
            # Whether it is a sulfur atom
            'is_s': [False for _ in range(num_atoms)],
            # Whether it is an oxygen atom
            'is_o': [False for _ in range(num_atoms)],
            # Whether it is a nitrogen atom
            'is_n': [False for _ in range(num_atoms)],
            'pair_to_bond_val': dict(),
            'ring_bonds': set()
        }

        # bookkeep atoms
        for j, atom in enumerate(mol.GetAtoms()):
            info['free_val'][j] += atom.GetTotalNumHs() + abs(atom.GetFormalCharge())
            # An aromatic carbon atom next to an aromatic nitrogen atom can get a
            # carbonyl b/c of bookkeeping of hydroxypyridines
            if atom.GetSymbol() == 'C':
                info['is_c'][j] = True
                if atom.GetIsAromatic():
                    for nbr in atom.GetNeighbors():
                        if nbr.GetSymbol() == 'N' and nbr.GetDegree() == 2:
                            info['is_c2_of_pyridine'][j] = True
                            break
            # A nitrogen atom should be allowed to become positively charged
            elif atom.GetSymbol() == 'N':
                info['free_val'][j] += 1 - atom.GetFormalCharge()
                info['is_n'][j] = True
            # Phosphorous atoms can form a phosphonium
            elif atom.GetSymbol() == 'P':
                info['free_val'][j] += 1 - atom.GetFormalCharge()
                info['is_p'][j] = True
            elif atom.GetSymbol() == 'O':
                info['is_o'][j] = True
            elif atom.GetSymbol() == 'S':
                info['is_s'][j] = True

        # bookkeep bonds
        for bond in mol.GetBonds():
            atom1, atom2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            atom1, atom2 = min(atom1, atom2), max(atom1, atom2)
            type_val = bond.GetBondTypeAsDouble()
            info['pair_to_bond_val'][(atom1, atom2)] = type_val
            if (atom1, atom2) in candidate_pairs:
                info['free_val'][atom1] += type_val
                info['free_val'][atom2] += type_val
            if bond.IsInRing():
                info['ring_bonds'].add((atom1, atom2))

        return info

    def bookkeep_product(self, mol):
        """Bookkeep reaction-related information of atoms/bonds in products

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance for products.

        Returns
        -------
        info : dict
            Reaction-related information of atoms/bonds in products
        """
        info = {
            'atoms': set()
        }
        for atom in mol.GetAtoms():
            info['atoms'].add(atom.GetAtomMapNum() - 1)

        return info

    def is_connected_change_combo(self, combo_ids, cand_change_adj):
        """Check whether the combo of bond changes yields a connected component.

        Parameters
        ----------
        combo_ids : tuple of int
            Ids for bond changes in the combination.
        cand_change_adj : bool ndarray of shape (N, N)
            Adjacency matrix for candidate bond changes. Two candidate bond
            changes are considered adjacent if they share a common atom.
            * N for the number of candidate bond changes.

        Returns
        -------
        bool
            Whether the combo of bond changes yields a connected component
        """
        if len(combo_ids) == 1:
            return True
        multi_hop_adj = np.linalg.matrix_power(
            cand_change_adj[combo_ids, :][:, combo_ids], len(combo_ids) - 1)
        # The combo is connected if the distance between
        # any pair of bond changes is within len(combo) - 1

        return np.all(multi_hop_adj)

    def is_valid_combo(self, combo_changes, reactant_info):
        """Whether the combo of bond changes is chemically valid.

        Parameters
        ----------
        combo_changes : list of 4-tuples
            Each tuple consists of atom1, atom2, type of bond change (in the form of related
            valence) and score for the change.
        reactant_info : dict
            Reaction-related information of reactants

        Returns
        -------
        bool
            Whether the combo of bond changes is chemically valid.
        """
        num_atoms = len(reactant_info['free_val'])
        force_even_parity = np.zeros((num_atoms,), dtype=bool)
        force_odd_parity = np.zeros((num_atoms,), dtype=bool)
        pair_seen = defaultdict(bool)
        free_val_tmp = reactant_info['free_val'].copy()
        for (atom1, atom2, change_type, score) in combo_changes:
            if pair_seen[(atom1, atom2)]:
                # A pair of atoms cannot have two types of changes. Even if we
                # randomly pick one, that will be reduced to a combo of less changed
                return False
            pair_seen[(atom1, atom2)] = True

            # Special valence rules
            atom1_type_val = atom2_type_val = change_type
            if change_type == 2:
                # to form a double bond
                if reactant_info['is_o'][atom1]:
                    if reactant_info['is_c2_of_pyridine'][atom2]:
                        atom2_type_val = 1.
                    elif reactant_info['is_p'][atom2]:
                        # don't count information of =o toward valence
                        # but require odd valence parity
                        atom2_type_val = 0.
                        force_odd_parity[atom2] = True
                    elif reactant_info['is_s'][atom2]:
                        atom2_type_val = 0.
                        force_even_parity[atom2] = True
                elif reactant_info['is_o'][atom2]:
                    if reactant_info['is_c2_of_pyridine'][atom1]:
                        atom1_type_val = 1.
                    elif reactant_info['is_p'][atom1]:
                        atom1_type_val = 0.
                        force_odd_parity[atom1] = True
                    elif reactant_info['is_s'][atom1]:
                        atom1_type_val = 0.
                        force_even_parity[atom1] = True
                elif reactant_info['is_n'][atom1] and reactant_info['is_p'][atom2]:
                    atom2_type_val = 0.
                    force_odd_parity[atom2] = True
                elif reactant_info['is_n'][atom2] and reactant_info['is_p'][atom1]:
                    atom1_type_val = 0.
                    force_odd_parity[atom1] = True
                elif reactant_info['is_p'][atom1] and reactant_info['is_c'][atom2]:
                    atom1_type_val = 0.
                    force_odd_parity[atom1] = True
                elif reactant_info['is_p'][atom2] and reactant_info['is_c'][atom1]:
                    atom2_type_val = 0.
                    force_odd_parity[atom2] = True

            reactant_pair_val = reactant_info['pair_to_bond_val'].get((atom1, atom2), None)
            if reactant_pair_val is not None:
                free_val_tmp[atom1] += reactant_pair_val - atom1_type_val
                free_val_tmp[atom2] += reactant_pair_val - atom2_type_val
            else:
                free_val_tmp[atom1] -= atom1_type_val
                free_val_tmp[atom2] -= atom2_type_val

        # False if 1) too many connections 2) sulfur valence not even
        # 3) phosphorous valence not odd
        if any(free_val_tmp < 0) or \
            any(aval % 2 != 0 for aval in free_val_tmp[force_even_parity]) or \
            any(aval % 2 != 1 for aval in free_val_tmp[force_odd_parity]):
            return False
        return True

    def edit_mol(self, reactant_mols, edits, product_info):
        """Simulate reaction via graph editing

        Parameters
        ----------
        reactant_mols : rdkit.Chem.rdchem.Mol
            RDKit molecule instances for reactants.
        edits : list of 4-tuples
            Bond changes for getting the product out of the reactants in a reaction.
            Each 4-tuple is of form (atom1, atom2, change_type, score), where atom1
            and atom2 are the end atoms to form or lose a bond, change_type is the
            type of bond change and score represents the confidence for the bond change
            by a model.
        product_info : dict
            proeduct_info['atoms'] gives a set of atom ids in the ground truth product molecule.

        Returns
        -------
        str
            SMILES for the main products
        """
        bond_change_to_type = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE,
                             3: Chem.rdchem.BondType.TRIPLE, 1.5: Chem.rdchem.BondType.AROMATIC}

        new_mol = Chem.RWMol(reactant_mols)
        [atom.SetNumExplicitHs(0) for atom in new_mol.GetAtoms()]

        for atom1, atom2, change_type, score in edits:
            bond = new_mol.GetBondBetweenAtoms(atom1, atom2)
            if bond is not None:
                new_mol.RemoveBond(atom1, atom2)
            if change_type > 0:
                new_mol.AddBond(atom1, atom2, bond_change_to_type[change_type])

        pred_mol = new_mol.GetMol()
        pred_smiles = Chem.MolToSmiles(pred_mol)
        pred_list = pred_smiles.split('.')
        pred_mols = []
        for pred_smiles in pred_list:
            mol = Chem.MolFromSmiles(pred_smiles)
            if mol is None:
                continue
            atom_set = set([atom.GetAtomMapNum() - 1 for atom in mol.GetAtoms()])
            if len(atom_set & product_info['atoms']) == 0:
                continue
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            pred_mols.append(mol)

        return '.'.join(sorted([Chem.MolToSmiles(mol) for mol in pred_mols]))

    def get_product_smiles(self, reactant_mols, edits, product_info):
        """Get the product smiles of the reaction

        Parameters
        ----------
        reactant_mols : rdkit.Chem.rdchem.Mol
            RDKit molecule instances for reactants.
        edits : list of 4-tuples
            Bond changes for getting the product out of the reactants in a reaction.
            Each 4-tuple is of form (atom1, atom2, change_type, score), where atom1
            and atom2 are the end atoms to form or lose a bond, change_type is the
            type of bond change and score represents the confidence for the bond change
            by a model.
        product_info : dict
            proeduct_info['atoms'] gives a set of atom ids in the ground truth product molecule.

        Returns
        -------
        str
            SMILES for the main products
        """
        smiles = self.edit_mol(reactant_mols, edits, product_info)
        if len(smiles) != 0:
            return smiles
        try:
            Chem.Kekulize(reactant_mols)
        except Exception as e:
            return smiles
        return self.edit_mol(reactant_mols, edits, product_info)

    def pre_process(self, num_candidate_bond_changes, max_num_changes_per_reaction,
                    max_num_change_combos_per_reaction):
        """Pre-process for the experiments.

        Parameters
        ----------
        num_candidate_bond_changes : int
            Number of candidate bond changes to consider for each ground truth reaction.
        max_num_changes_per_reaction : int
            Maximum number of bond changes per reaction.
        max_num_change_combos_per_reaction : int
            Number of bond change combos to consider for each reaction.

        Returns
        -------
        valid_candidate_combos : list of list
            valid_candidate_combos[i] gives combos of bond changes for the i-th reaction.
            valid_candidate_combos[i][j] gives the j-th combo for the i-th reaction, which
            is of form ``atom1, atom2, change_type, score``, where ``atom1`` and ``atom2`` are
            end atoms for the bond to form/break, ``change_type`` is the type for the bond change
            and ``score`` is the confidence in the bond change by a model.
        """
        all_valid_candidate_combos = []
        for i in range(len(self.reactant_mols)):
            candidate_pairs = [(atom1, atom2) for (atom1, atom2, _, _)
                               in self.candidate_bond_changes[i]]
            reactant_mol = self.reactant_mols[i]
            reactant_info = self.bookkeep_reactant(reactant_mol, candidate_pairs)
            product_mol = self.product_mols[i]
            if self.train_mode:
                product_info = self.bookkeep_product(product_mol)

            # Filter out candidate new bonds already in reactants
            candidate_bond_changes = []
            for (atom1, atom2, change_type, score) in self.candidate_bond_changes[i]:
                if ((atom1, atom2) not in reactant_info['pair_to_bond_val']) or \
                    (reactant_info['pair_to_bond_val'][(atom1, atom2)] != change_type):
                    candidate_bond_changes.append((atom1, atom2, change_type, score))
            candidate_bond_changes = candidate_bond_changes[:num_candidate_bond_changes]

            # Check if two bond changes have atom in common
            cand_change_adj = np.eye(len(candidate_bond_changes), dtype=bool)
            for id1, candidate1 in enumerate(candidate_bond_changes):
                atom1_1, atom1_2, _, _ = candidate1
                for id2, candidate2 in enumerate(candidate_bond_changes):
                    atom2_1, atom2_2, _, _ = candidate2
                    if atom1_1 == atom2_1 or atom1_1 == atom2_2 or \
                        atom1_2 == atom2_1 or atom1_2 == atom2_2:
                        cand_change_adj[id1, id2] = cand_change_adj[id2, id1] = True

            # Enumerate combinations of k candidate bond changes and record
            # those that are connected and chemically valid
            valid_candidate_combos = []
            cand_change_ids = range(len(candidate_bond_changes))
            for k in range(1, max_num_changes_per_reaction + 1):
                for combo_ids in combinations(cand_change_ids, k):
                    # Check if the changed bonds form a connected component
                    if not self.is_connected_change_combo(combo_ids, cand_change_adj):
                        continue
                    combo_changes = [candidate_bond_changes[i] for i in combo_ids]
                    # Check if the combo is chemically valid
                    if self.is_valid_combo(combo_changes, reactant_info):
                        valid_candidate_combos.append(combo_changes)

            if self.train_mode:
                random.shuffle(valid_candidate_combos)
                # Index for the combo of candidate bond changes
                # that is equivalent to the gold combo
                real_combo_id = -1
                for i, combo_changes in enumerate(valid_candidate_combos):
                    if set([(atom1, atom2, change_type) for
                            (atom1, atom2, change_type, score) in combo_changes]) == \
                        set(self.real_bond_changes[i]):
                        real_combo_id = i
                        break

                # If we fail to find the real combo, make it the first entry
                if real_combo_id == -1:
                    valid_candidate_combos = \
                        [[(atom1, atom2, change_type, 0.0)
                          for (atom1, atom2, change_type) in self.real_bond_changes[i]]] + \
                        valid_candidate_combos
                else:
                    valid_candidate_combos[0], valid_candidate_combos[real_combo_id] = \
                        valid_candidate_combos[real_combo_id], valid_candidate_combos[0]

                product_smiles = self.get_product_smiles(
                    reactant_mol, valid_candidate_combos[0], product_info)
                if len(product_smiles) > 0:
                    # Remove combos yielding duplicate products
                    product_smiles = set([product_smiles])
                    new_candidate_combos = [valid_candidate_combos[0]]

                    for combo in valid_candidate_combos[1:]:
                        smiles = self.get_product_smiles(reactant_mol, combo, product_info)
                        if smiles in product_smiles or len(smiles) == 0:
                            continue
                        product_smiles.append(smiles)
                        new_candidate_combos.append(combo)
                else:
                    print('\nwarning! could not recover true smiles from gbonds: {}'.format(
                        Chem.MolToSmiles(product_mol)))
                    print('reactant smiles: {} graph edits: {}'.format(
                        Chem.MolToSmiles(reactant_mol), self.real_bond_changes[i]))

            valid_candidate_combos = valid_candidate_combos[:max_num_change_combos_per_reaction]
            all_valid_candidate_combos.append(valid_candidate_combos)

        return all_valid_candidate_combos

    def ignore_large(self, ignore=True):
        """Whether to ignore reactions where reactants contain too many atoms.

        Parameters
        ----------
        ignore : bool
            If ``ignore``, reactions where reactants contain too many atoms will be ignored.
        """
        self.ignore_large_samples = ignore

    def __len__(self):
        """Get the size for the dataset.

        Returns
        -------
        int
            Number of reactions in the dataset.
        """
        if self.ignore_large_samples:
            return len(self.ids_for_small_samples)
        else:
            return len(self.reactant_mols)

    def __getitem__(self, item):
        """Get the i-th datapoint.

        Parameters
        ----------
        item : int
            Index for the datapoint.
        """
        if self.ignore_large_samples:
            item = self.ids_for_small_samples[item]
        return NotImplementedError

class USPTORank(WLNRankDataset):
    """USPTO dataset for ranking candidate products.

    The dataset contains reactions from patents granted by United States Patent
    and Trademark Office (USPTO), collected by Lowe [1]. Jin et al. removes duplicates
    and erroneous reactions, obtaining a set of 480K reactions. They divide it
    into 400K, 40K, and 40K for training, validation and test.

    References:

        * [1] Patent reaction extraction
        * [2] Predicting Organic Reaction Outcomes with Weisfeiler-Lehman Network

    Parameters
    ----------
    subset : str
        Whether to use the training/validation/test set as in Jin et al.

        * 'train' for the training set
        * 'val' for the validation set
        * 'test' for the test set
    candidate_bond_path : str
        Path to the candidate bond changes for product enumeration, where each line is
        candidate bond changes for a reaction by a WLN for reaction center prediction.
    size_cutoff : int
        By calling ``.ignore_large(True)``, we can optionally ignore reactions whose reactants
        contain more than ``size_cutoff`` atoms. Default to 100.
    max_num_changes_per_reaction : int
        Maximum number of bond changes per reaction. Default to 5.
    log_every : int
        Print a progress update every time ``log_every`` reactions are pre-processed.
        Default to 10000.
    load : bool
        Whether to load the previously pre-processed dataset or pre-process from scratch.
        ``load`` should be False when we want to try different graph construction and
        featurization methods and need to preprocess from scratch. Default to True.
    """
    def __init__(self,
                 subset,
                 candidate_bond_path,
                 size_cutoff=100,
                 max_num_changes_per_reaction=5,
                 log_every=10000,
                 load=True):
        assert subset in ['train', 'val', 'test'], \
            'Expect subset to be "train" or "val" or "test", got {}'.format(subset)
        print('Preparing {} subset of USPTO for product candidate ranking.'.format(subset))
        self._subset = subset
        if subset == 'val':
            subset = 'valid'

        self._url = 'dataset/uspto.zip'
        data_path = get_download_dir() + '/uspto.zip'
        extracted_data_path = get_download_dir() + '/uspto'
        download(_get_dgl_url(self._url), path=data_path)
        extract_archive(data_path, extracted_data_path)

        if self.subset == 'train':
            train_mode = True
        else:
            train_mode = False

        super(USPTORank, self).__init__(
            raw_file_path=extracted_data_path + '/{}.txt.proc'.format(subset),
            candidate_bond_path=candidate_bond_path,
            mol_graph_path=extracted_data_path + '/{}_mol_graphs.bin'.format(subset),
            size_cutoff=size_cutoff,
            max_num_changes_per_reaction=max_num_changes_per_reaction,
            train_mode=train_mode,
            log_every=log_every,
            load=load)

    @property
    def subset(self):
        """Get the subset used for USPTOCenter

        Returns
        -------
        str

            * 'full' for the complete dataset
            * 'train' for the training set
            * 'val' for the validation set
            * 'test' for the test set
        """
        return self._subset
