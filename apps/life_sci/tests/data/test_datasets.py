import os

from dgllife.data import *
from dgllife.data.uspto import get_bond_changes, process_file

def remove_file(fname):
    if os.path.isfile(fname):
        try:
            os.remove(fname)
        except OSError:
            pass

def test_pubchem_aromaticity():
    print('Test pubchem aromaticity')
    dataset = PubChemBioAssayAromaticity()
    remove_file('pubchem_aromaticity_dglgraph.bin')

def test_tox21():
    print('Test Tox21')
    dataset = Tox21()
    remove_file('tox21_dglgraph.bin')

def test_alchemy():
    print('Test Alchemy')
    dataset = TencentAlchemyDataset(mode='valid',
                                    node_featurizer=None,
                                    edge_featurizer=None)
    dataset = TencentAlchemyDataset(mode='valid',
                                    node_featurizer=None,
                                    edge_featurizer=None,
                                    load=False)

def test_pdbbind():
    print('Test PDBBind')
    dataset = PDBBind(subset='core', remove_hs=True)

def test_wln_reaction():
    print('Test datasets for reaction prediction with WLN')

    reaction1 = '[CH2:15]([CH:16]([CH3:17])[CH3:18])[Mg+:19].[CH2:20]1[O:21][CH2:22][CH2:23]' \
                '[CH2:24]1.[Cl-:14].[OH:1][c:2]1[n:3][cH:4][c:5]([C:6](=[O:7])[N:8]([O:9]' \
                '[CH3:10])[CH3:11])[cH:12][cH:13]1>>[OH:1][c:2]1[n:3][cH:4][c:5]([C:6](=[O:7])' \
                '[CH2:15][CH:16]([CH3:17])[CH3:18])[cH:12][cH:13]1\n'
    reaction2 = '[CH3:14][NH2:15].[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])' \
                '[cH:10][cH:11][c:12]1[Cl:13].[OH2:16]>>[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]' \
                '([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[NH:15][CH3:14]\n'
    reactions = [reaction1, reaction2]

    # Test utility functions
    assert get_bond_changes(reaction2) == {('12', '13', 0.0), ('12', '15', 1.0)}
    with open('test.txt', 'w') as f:
        for reac in reactions:
            f.write(reac)
    process_file('test.txt')
    with open('test.txt.proc', 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            l = lines[i].strip()
            react = reactions[i].strip()
            bond_changes = get_bond_changes(react)
            assert l == '{} {}'.format(
                react,
                ';'.join(['{}-{}-{}'.format(x[0], x[1], x[2]) for x in bond_changes]))
    remove_file('test.txt.proc')

    # Test configured dataset
    dataset = WLNReactionDataset('test.txt', 'test_graphs.bin')
    remove_file('test.txt')
    remove_file('test.txt.proc')
    remove_file('test_graphs.bin')

if __name__ == '__main__':
    test_pubchem_aromaticity()
    test_tox21()
    test_alchemy()
    test_pdbbind()
    test_wln_reaction()
