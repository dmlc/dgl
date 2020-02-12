import pandas as pd
import sys

from .csv_dataset import MoleculeCSVDataset
from ..utils import smiles_to_bigraph
from ...utils import get_download_dir, download, _get_dgl_url
from ....base import dgl_warning
from ....contrib.deprecation import deprecated

class PubChemBioAssayAromaticity(MoleculeCSVDataset):
    """Subset of PubChem BioAssay Dataset for aromaticity prediction.

    The dataset was constructed in `Pushing the Boundaries of Molecular Representation for Drug
    Discovery with the Graph Attention Mechanism.
    <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__ and is accompanied by the task of predicting
    the number of aromatic atoms in molecules.

    The dataset was constructed by sampling 3945 molecules with 0-40 aromatic atoms from the
    PubChem BioAssay dataset.

    Parameters
    ----------
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
        featurization methods and need to pre-process from scratch. Default to True.
    """
    @deprecated('Import PubChemBioAssayAromaticity from dgllife.data instead.', 'class')
    def __init__(self, smiles_to_graph=smiles_to_bigraph,
                 node_featurizer=None, edge_featurizer=None, load=True):
        if 'pandas' not in sys.modules:
            dgl_warning("Please install pandas")

        self._url = 'dataset/pubchem_bioassay_aromaticity.csv'
        data_path = get_download_dir() + '/pubchem_bioassay_aromaticity.csv'
        download(_get_dgl_url(self._url), path=data_path)
        df = pd.read_csv(data_path)

        super(PubChemBioAssayAromaticity, self).__init__(
            df, smiles_to_graph, node_featurizer, edge_featurizer, "cano_smiles",
            "pubchem_aromaticity_dglgraph.bin", load=load)
