"""USPTO for reaction prediction"""
import random
import torch

from collections import defaultdict
from dgl.data.utils import get_download_dir, download, _get_dgl_url, extract_archive
from functools import partial
from multiprocessing import Pool
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolops

from ..utils.mol_to_graph import mol_to_bigraph

__all__ = ['USPTO']

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

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

def preprocess_single_reaction_data(reaction_data, mol_to_graph, node_featurizer,
                                    edge_featurizer, atom_pair_featurizer):
    """Construct DGLGraphs and featurize them for a single reaction data.

    Parameters
    ----------
    reaction_data : 3-tuple
        The first element in the tuple stands for RDKit molecule instance of the reactants.
        The second element in the tuple stands for reaction. The third element in the
        tuple stands for graph edits in the reaction.
    mol_to_graph: callable, str -> DGLGraph
        A function turning RDKit molecule instances into DGLGraphs.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph.
    atom_pair_featurizer : None or callable, str -> dict
        If not None, featurization for each pair of atoms in multiple reactants. The result
        will be used to update edata in the complete DGLGraphs.

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance
    reactant_mol_graph : DGLGraph
        DGLGraph for the reactants
    atom_pair_features : None or float32 tensor of shape (V^2, dim)
        If not None, features for each pair of atoms in multiple reactants.
        V for the number of atoms in the reactants.
    label : float32 tensor of shape (V^2, 5)
        Labels constructed for each pair of atoms in multiple reactants.
    """
    mol = reaction_data[0]
    reaction = reaction_data[1]
    graph_edits = reaction_data[2]
    reactants = reaction.split('>')[0]
    reactant_mol_graph = mol_to_graph(mol, node_featurizer=node_featurizer,
                                      edge_featurizer=edge_featurizer,
                                      canonical_atom_order=False)
    if atom_pair_featurizer is not None:
        atom_pair_features = atom_pair_featurizer(reactants)
    else:
        atom_pair_features = None
    label = get_pair_label(mol, graph_edits)

    return mol, reactant_mol_graph, atom_pair_features, label

def preprocess_reaction_data(all_reaction_data, mol_to_graph, node_featurizer,
                             edge_featurizer, atom_pair_featurizer, num_processes, load):
    """Construct DGLGraphs and featurize them.

    Parameters
    ----------
    all_reaction_data : list of 3-tuples
        The first element in the tuple stands for RDKit molecule instance of the reactants.
        The second element in the tuple stands for reaction. The third element in the
        tuple stands for graph edits in the reaction.
    mol_to_graph: callable, str -> DGLGraph
        A function turning RDKit molecule instances into DGLGraphs.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph.
    atom_pair_featurizer : callable, str -> dict
        Featurization for each pair of atoms in multiple reactants. The result will be
        used to update edata in the complete DGLGraphs.
    num_processes : int
        Number of processes for multiprocessing.
    load : bool
        Whether to load the previously pre-processed dataset or pre-process from scratch.
        ``load`` should be False when we want to try different graph construction and
        featurization methods and need to preprocess from scratch.

    Returns
    -------
    mols : list of rdkit.Chem.rdchem.Mol
        RDKit molecules for reactants.
    reactant_mol_graphs : list of DGLGraphs
        DGLGraphs for reactants.
    atom_pair_features : list of None or float32 tensor of shape (V_i^2, dim)
        If not None, features for each pair of atoms in multiple reactants.
        V_i for the number of atoms in the i-th reactants.
    labels : list of float32 tensor of shape (V_i^2, 5)
        Labels constructed for each pair of atoms in multiple reactants.
    """
    # Todo: check whether the storage files exist
    if not load:
        return NotImplementedError
    else:
        with Pool(processes=num_processes) as pool:
            results = pool.map_async(partial(
                preprocess_single_reaction_data, mol_to_graph=mol_to_graph,
                node_featurizer=node_featurizer, edge_featurizer=edge_featurizer,
                atom_pair_featurizer=atom_pair_featurizer), all_reaction_data)
            results = results.get()
        mols, reactant_mol_graphs, atom_pair_features, labels = map(list, zip(*results))
    return mols, reactant_mol_graphs, atom_pair_features, labels

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
        ndata for a DGLGraph. Default to None.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.
    atom_pair_featurizer : callable, str -> dict
        Featurization for each pair of atoms in multiple reactants. The result will be
        used to update edata in the complete DGLGraphs.
    num_processes : int
        Number of processes for multiprocessing. Default to 8.
    load : bool
        Whether to load the previously pre-processed dataset or pre-process from scratch.
        ``load`` should be False when we want to try different graph construction and
        featurization methods and need to preprocess from scratch. Default to True.
    """
    def __init__(self,
                 subset,
                 mol_to_graph=mol_to_bigraph,
                 node_featurizer=None,
                 edge_featurizer=None,
                 atom_pair_featurizer=None,
                 num_processes=8,
                 load=True):
        super(USPTO, self).__init__()

        assert subset in ['full', 'train', 'val', 'test'], \
            'Expect subset to be "train" or "val" or "test", got {}'.format(subset)
        self._subset = subset

        self._url = 'dataset/uspto.zip'
        data_path = get_download_dir() + '/uspto.zip'
        extracted_data_path = get_download_dir() + '/uspto'
        download(_get_dgl_url(self._url), path=data_path)
        extract_archive(data_path, extracted_data_path)

        self.mols = []
        self.reactant_mol_graphs = []
        self.atom_pair_features = []
        self.labels = []

        if self.subset == 'full':
            subsets = ['train', 'valid', 'test']
        elif self.subset == 'val':
            subsets = ['valid']
        else:
            subsets = [self.subset]

        for set in subsets:
            print('Preparing {} set'.format(set))
            file_path = extracted_data_path + '/{}.txt.proc'.format(set)
            all_reaction_data = self.load_reaction_data(file_path)

            # Todo: remove the line below
            all_reaction_data = all_reaction_data[:1000]
            mols, reactant_mol_graphs, atom_pair_features, labels = preprocess_reaction_data(
                all_reaction_data,  mol_to_graph, node_featurizer,
                edge_featurizer, atom_pair_featurizer, num_processes, load)
            self.mols.extend(mols)
            self.reactant_mol_graphs.extend(reactant_mol_graphs)
            self.atom_pair_features.extend(atom_pair_features)
            self.labels.extend(labels)

    def load_reaction_data(self, file_path):
        """Load reaction data from the raw file.

        Parameters
        ----------
        file_path : str
            Path to read the file.

        Returns
        -------
        list of 3-tuples
            The first element in the tuple stands for RDKit molecule instance of the reactants.
            The second element in the tuple stands for reaction. The third element in the
            tuple stands for graph edits in the reaction.
        """
        all_reaction_data = []
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
                # Reorder atoms according to the order specified in the atom map
                atom_map_order = [-1 for _ in range(mol.GetNumAtoms())]
                for i in range(mol.GetNumAtoms()):
                    atom = mol.GetAtomWithIdx(i)
                    atom_map_order[atom.GetIntProp('molAtomMapNumber') - 1] = i
                mol = rdmolops.RenumberAtoms(mol, atom_map_order)
                if mol is not None:
                    all_reaction_data.append((mol, reaction, graph_edits))

        random.shuffle(all_reaction_data)

        return all_reaction_data

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
        rdkit.Chem.rdchem.Mol
            RDKit molecule instance
        DGLGraph
            DGLGraph for the ith molecular graph
        DGLGraph
            Complete DGLGraph, which will be needed for predicting
            scores between each pair of atoms
        float32 tensor of shape (V^2, 5)
            Labels for reaction center prediction.
            V for the number of atoms in the reactants.
        """
        return self.mols[item], self.reactant_mol_graphs[item], \
               self.reactant_complete_graphs[item], self.labels[item]
