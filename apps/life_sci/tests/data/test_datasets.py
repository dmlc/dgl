import os

from dgllife.data import *

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

if __name__ == '__main__':
    test_pubchem_aromaticity()
    test_tox21()
    test_alchemy()
    test_pdbbind()
