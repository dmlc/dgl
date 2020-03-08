"""USPTO for reaction prediction"""
import numpy as np
import torch

from collections import defaultdict
from dgl.data.utils import get_download_dir, download, _get_dgl_url, extract_archive, \
    save_graphs, load_graphs
from functools import partial
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolops

from ..utils.featurizers import BaseAtomFeaturizer, ConcatFeaturizer, atom_type_one_hot, \
    atom_degree_one_hot, atom_explicit_valence_one_hot, atom_implicit_valence_one_hot, \
    atom_is_aromatic, BaseBondFeaturizer, bond_type_one_hot, bond_is_conjugated, bond_is_in_ring
from ..utils.mol_to_graph import mol_to_bigraph, mol_to_complete_graph

__all__ = ['USPTO']

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

class USPTO(object):
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
        Whether to use the full dataset or training/validation/test set as in Jin et al.

        * 'full' for the complete dataset
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
        super(USPTO, self).__init__()

        assert subset in ['full', 'train', 'val', 'test'], \
            'Expect subset to be "train" or "val" or "test", got {}'.format(subset)
        self._subset = subset

        self._atom_pair_featurizer = atom_pair_featurizer
        self._url = 'dataset/uspto.zip'
        data_path = get_download_dir() + '/uspto.zip'
        extracted_data_path = get_download_dir() + '/uspto'
        download(_get_dgl_url(self._url), path=data_path)
        extract_archive(data_path, extracted_data_path)

        self.reactions = []
        self.graph_edits = []
        self.mols = []
        self.reactant_mol_graphs = []
        self.atom_pair_features = []
        self.atom_pair_labels = []
        # Map number of nodes to a corresponding complete graph
        self.complete_graphs = dict()

        if self.subset == 'full':
            subsets = ['train', 'valid', 'test']
        elif self.subset == 'val':
            subsets = ['valid']
        else:
            subsets = [self.subset]

        for set in subsets:
            print('Preparing {} set'.format(set))
            file_path = extracted_data_path + '/{}.txt.proc'.format(set)
            set_mols, set_reactions, set_graph_edits = self.load_reaction_data(file_path)
            self.mols.extend(set_mols)
            self.reactions.extend(set_reactions)
            self.graph_edits.extend(set_graph_edits)
            set_reactant_mol_graphs = []
            if load:
                set_reactant_mol_graphs, _ = load_graphs(
                    extracted_data_path + '/{}_mol_graphs.bin'.format(set))
            else:
                for i in range(len(set_mols)):
                    print('Processing reaction {:d}/{:d} for {} set'.format(
                        i + 1, len(set_mols), set))
                    mol = set_mols[i]
                    reactant_mol_graph = mol_to_graph(mol, node_featurizer=node_featurizer,
                                                      edge_featurizer=edge_featurizer,
                                                      canonical_atom_order=False)
                    set_reactant_mol_graphs.append(reactant_mol_graph)

                save_graphs(extracted_data_path + '/{}_mol_graphs.bin'.format(set),
                            set_reactant_mol_graphs)

            self.mols.extend(set_mols)
            self.reactant_mol_graphs.extend(set_reactant_mol_graphs)
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

    @property
    def subset(self):
        """Get the subset used for USPTO

        Returns
        -------
        str

            * 'full' for the complete dataset
            * 'train' for the training set
            * 'val' for the validation set
            * 'test' for the test set
        """
        return self._subset

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
