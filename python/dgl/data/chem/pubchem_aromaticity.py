import pandas as pd
import sys

from .csv_dataset import MoleculeCSVDataset
from .utils import smiles_to_bigraph
from ..utils import get_download_dir, download, _get_dgl_url

class PubChemBioAssayAromaticity(MoleculeCSVDataset):
    """Subset of PubChem BioAssay Dataset for aromaticity prediction.

    The dataset was constructed in `Pushing the Boundaries of Molecular Representation for Drug
    Discovery with the Graph Attention Mechanism.
    <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__ and is accompanied by the task of predicting
    the number of aromatic atoms in molecules.

    The dataset was constructed by sampling 3945 molecules with 0-40 aromatic atoms from the
    PubChem BioAssay dataset.
    """
    def __init__(self, smiles_to_graph=smiles_to_bigraph,
                 atom_featurizer=None,
                 bond_featurizer=None):
        if 'pandas' not in sys.modules:
            from ...base import dgl_warning
            dgl_warning("Please install pandas")

        self._url = 'dataset/pubchem_bioassay_aromaticity.csv'
        data_path = get_download_dir() + '/pubchem_bioassay_aromaticity.csv'
        download(_get_dgl_url(self._url), path=data_path)
        df = pd.read_csv(data_path)

        super(PubChemBioAssayAromaticity, self).__init__(df, smiles_to_graph, atom_featurizer, bond_featurizer,
                                                         "cano_smiles", "pubchem_aromaticity_dglgraph.bin")
