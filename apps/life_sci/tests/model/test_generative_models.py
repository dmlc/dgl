import torch

from rdkit import Chem

from dgllife.model import DGMG, DGLJTNNVAE

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
    assert model(rdkit_mol=True) is not None

    model = DGMG(atom_types=['O', 'Cl', 'C', 'S', 'F', 'Br', 'N'],
                 bond_types=[Chem.rdchem.BondType.SINGLE,
                             Chem.rdchem.BondType.DOUBLE,
                             Chem.rdchem.BondType.TRIPLE])
    assert model(
        actions=[(0, 2), (1, 3), (0, 0), (1, 0), (2, 0), (1, 3), (0, 7)], rdkit_mol=True) == 'CO'
    assert model(rdkit_mol=False) is None
    model.eval()
    assert model(rdkit_mol=True) is not None

def test_jtnn():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model = DGLJTNNVAE(hidden_size=1,
                       latent_size=2,
                       depth=1).to(device)

if __name__ == '__main__':
    test_dgmg()
    test_jtnn()
