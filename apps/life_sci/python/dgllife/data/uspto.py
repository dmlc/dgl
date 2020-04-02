"""USPTO for reaction prediction"""
import numpy as np
import os
import torch

from collections import defaultdict
from dgl.data.utils import get_download_dir, download, _get_dgl_url, extract_archive, \
    save_graphs, load_graphs
from functools import partial
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolops
from tqdm import tqdm

from ..utils.featurizers import BaseAtomFeaturizer, ConcatFeaturizer, atom_type_one_hot, \
    atom_degree_one_hot, atom_explicit_valence_one_hot, atom_implicit_valence_one_hot, \
    atom_is_aromatic, BaseBondFeaturizer, bond_type_one_hot, bond_is_conjugated, bond_is_in_ring
from ..utils.mol_to_graph import mol_to_bigraph, mol_to_complete_graph

__all__ = ['WLNReactionDataset',
           'USPTO']

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Atom types distinguished in featurization
atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
              'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
              'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
              'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi',
              'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs']

default_node_featurizer = BaseAtomFeaturizer({
    'hv': ConcatFeaturizer(
        [partial(atom_type_one_hot, allowable_set=atom_types, encode_unknown=True),
         partial(atom_degree_one_hot, allowable_set=list(range(6))),
         atom_explicit_valence_one_hot,
         partial(atom_implicit_valence_one_hot, allowable_set=list(range(6))),
         atom_is_aromatic]
    )
})

default_edge_featurizer = BaseBondFeaturizer({
    'he': ConcatFeaturizer([
        bond_type_one_hot, bond_is_conjugated, bond_is_in_ring]
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

class WLNReactionDataset(object):
    """Dataset for reaction prediction with WLN

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
    """
    def __init__(self,
                 raw_file_path,
                 mol_graph_path,
                 mol_to_graph=mol_to_bigraph,
                 node_featurizer=default_node_featurizer,
                 edge_featurizer=default_edge_featurizer,
                 atom_pair_featurizer=default_atom_pair_featurizer,
                 load=True):
        super(WLNReactionDataset, self).__init__()

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
            self.load_reaction_data(path_to_reaction_file)
        if load and os.path.isfile(mol_graph_path):
            self.reactant_mol_graphs, _ = load_graphs(mol_graph_path)
        else:
            self.reactant_mol_graphs = []
            for i in range(len(full_mols)):
                if i % 10000 == 0:
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

    def load_reaction_data(self, file_path):
        """Load reaction data from the raw file.

        Parameters
        ----------
        file_path : str
            Path to read the file.

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
                if i % 10000 == 0:
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
            Reaction.
        str
            Graph edits for the reaction
        rdkit.Chem.rdchem.Mol
            RDKit molecule instance for reactants
        DGLGraph
            DGLGraph for the ith molecular graph of reactants
        DGLGraph
            Complete DGLGraph for reactants, which will be needed for predicting
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

class USPTO(WLNReactionDataset):
    """USPTO dataset for reaction prediction.

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
                 node_featurizer=default_node_featurizer,
                 edge_featurizer=default_edge_featurizer,
                 atom_pair_featurizer=default_atom_pair_featurizer,
                 load=True):
        assert subset in ['train', 'val', 'test'], \
            'Expect subset to be "train" or "val" or "test", got {}'.format(subset)
        print('Preparing {} subset of USPTO'.format(subset))
        self._subset = subset
        if subset == 'val':
            subset = 'valid'

        self._url = 'dataset/uspto.zip'
        data_path = get_download_dir() + '/uspto.zip'
        extracted_data_path = get_download_dir() + '/uspto'
        download(_get_dgl_url(self._url), path=data_path)
        extract_archive(data_path, extracted_data_path)

        super(USPTO, self).__init__(
            raw_file_path=extracted_data_path + '/{}.txt'.format(subset),
            mol_graph_path=extracted_data_path + '/{}_mol_graphs.bin'.format(subset),
            mol_to_graph=mol_to_graph,
            node_featurizer=node_featurizer,
            edge_featurizer=edge_featurizer,
            atom_pair_featurizer=atom_pair_featurizer,
            load=load)

    @property
    def subset(self):
        """Get the subset used for USPTO

        Returns
        -------
        str

            * 'train' for the training set
            * 'val' for the validation set
            * 'test' for the test set
        """
        return self._subset
