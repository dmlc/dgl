import copy

import rdkit.Chem as Chem


def get_slots(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [
        (atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs())
        for atom in mol.GetAtoms()
    ]


class Vocab(object):
    def __init__(self, smiles_list):
        self.vocab = smiles_list
        self.vmap = {x: i for i, x in enumerate(self.vocab)}
        self.slots = [get_slots(smiles) for smiles in self.vocab]

    def get_index(self, smiles):
        return self.vmap[smiles]

    def get_smiles(self, idx):
        return self.vocab[idx]

    def get_slots(self, idx):
        return copy.deepcopy(self.slots[idx])

    def size(self):
        return len(self.vocab)
