import os
import pickle

from dgl.data.utils import download, _get_dgl_url
from rdkit import Chem

from dglls.model import DGMG, DGLJTNNVAE

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

def remove_file(fname):
    if os.path.isfile(fname):
        try:
            os.remove(fname)
        except OSError:
            pass

def test_jtnn():
    url = _get_dgl_url('dglls/jtnn_test_batch.pkl')
    local_path = 'jtnn_test_batch.pkl'
    download(url, path=local_path)
    with open(local_path, 'rb') as f:
        batch = pickle.load(f)

    beta = 1.0
    model = DGLJTNNVAE(hidden_size=1,
                       latent_size=2,
                       depth=1)
    model(batch, beta)

    remove_file(local_path)

if __name__ == '__main__':
    test_dgmg()
    test_jtnn()
