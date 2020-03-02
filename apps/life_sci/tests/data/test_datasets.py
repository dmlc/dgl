import os

from dgllife.data import *
from dgllife.utils import mol_to_bigraph

def remove_file(fname):
    if os.path.isfile(fname):
        try:
            os.remove(fname)
        except OSError:
            pass

def test_pubchem_aromaticity():
    dataset = PubChemBioAssayAromaticity()
    remove_file('pubchem_aromaticity_dglgraph.bin')

def test_tox21():
    dataset = Tox21()
    remove_file('tox21_dglgraph.bin')

def test_alchemy():
    dataset = TencentAlchemyDataset(mode='valid',
                                    node_featurizer=None,
                                    edge_featurizer=None)
    dataset = TencentAlchemyDataset(mode='valid',
                                    node_featurizer=None,
                                    edge_featurizer=None,
                                    load=False)

def test_pdbbind():
    dataset = PDBBind(subset='core', remove_hs=True)

def test_uspto():
    dataset = USPTO(subset='full')
    dataset = USPTO(subset='train')
    dataset = USPTO(subset='val')
    dataset = USPTO(subset='test')

if __name__ == '__main__':
    test_pubchem_aromaticity()
    test_tox21()
    test_alchemy()
    test_pdbbind()
    test_uspto()
