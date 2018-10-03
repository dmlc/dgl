from torch.utils.data import Dataset
import numpy as np

class DGLMoleculeDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from .mol_tree_nx import DGLMolTree
        smiles = self.data[idx]
        mol_tree = DGLMolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()
        return mol_tree
