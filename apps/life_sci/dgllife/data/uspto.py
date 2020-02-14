"""USPTO for reaction prediction"""
import random

from dgl.data.utils import get_download_dir, download, _get_dgl_url, extract_archive
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolops

from ..utils.mol_to_graph import mol_to_bigraph

__all__ = ['USPTO']

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

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
    smiles_to_graph: callable, str -> DGLGraph
        A function turning smiles into a DGLGraph.
        Default to :func:`dgl.data.chem.smiles_to_bigraph`.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.
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
                 load=True):
        super(USPTO, self).__init__()

        assert subset in ['full', 'train', 'val', 'test'], \
            'Expect subset to be "train" or "val" or "test", got {}'.format(subset)
        self._subset = subset
        self._mol_to_graph = mol_to_graph
        self._node_featurizer = node_featurizer
        self._edge_featurizer = edge_featurizer
        self._load = load

        self._url = 'dataset/uspto.zip'
        data_path = get_download_dir() + '/uspto.zip'
        extracted_data_path = get_download_dir() + '/uspto'
        download(_get_dgl_url(self._url), path=data_path)
        extract_archive(data_path, extracted_data_path)

        self.reactant_graphs = []

        if self.subset in ['full', 'train']:
            self.preprocess(extracted_data_path + '/train.txt.proc')
        if self.subset in ['full', 'val']:
            self.preprocess(extracted_data_path + '/valid.txt.proc')
        if self.subset in ['full', 'test']:
            self.preprocess(extracted_data_path + '/test.txt.proc')

    def preprocess(self, file_path):
        """Read the raw data file and pre-process it.

        During the pre-processing, DGLGraphs are constructed with nodes and edges featurized.

        Parameters
        ----------
        file_path : str
            Path to read the file.
        """
        all_reaction_data = []
        with open(file_path, 'r') as f:
            for line in f:
                # Each line represents a reaction and the corresponding graph edits
                reaction, graph_edits = line.strip("\r\n ").split()
                all_reaction_data.append((reaction, graph_edits))

        random.shuffle(all_reaction_data)
        for reaction, graph_edits in all_reaction_data:
            reactants = reaction.split('>>')[0]
            mol = Chem.MolFromSmiles(reactants)
            # Reorder atoms according to the order specified in the atom map
            atom_map_order = [-1 for _ in range(mol.GetNumAtoms())]
            for i in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(i)
                atom_map_order[atom.GetIntProp('molAtomMapNumber') - 1] = i
            mol = rdmolops.RenumberAtoms(mol, atom_map_order)
            graph = self._mol_to_graph(mol, node_featurizer=self._node_featurizer,
                                       edge_featurizer=self._edge_featurizer,
                                       canonical_atom_order=False)
            self.reactant_graphs.append(graph)

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
