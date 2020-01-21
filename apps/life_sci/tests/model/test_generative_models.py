from rdkit import Chem

from dglls.model import DGMG

def test_dgmg():
    model = DGMG(atom_types=['O', 'Cl', 'C', 'S', 'F', 'Br', 'N'],
                 bond_types=[Chem.rdchem.BondType.SINGLE,
                             Chem.rdchem.BondType.DOUBLE,
                             Chem.rdchem.BondType.TRIPLE],
                 node_hidden_size=1,
                 num_prop_rounds=1,
                 dropout=0.2)
    assert model(
        actions=[(0, 2), (1, 3), (0, 0), (1, 0), (2, 0), (1, 3), (0, 7)], rdkit_mol=True) == 'CO'
    assert model(rdkit_mol=False) is None
    model.eval()
    model(rdkit_mol=True)

if __name__ == '__main__':
    test_dgmg()
