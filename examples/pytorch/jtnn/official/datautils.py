from torch.utils.data import Dataset
import numpy as np

class MoleculeDataset(Dataset):

    def __init__(self, data_file):
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        from .mol_tree import MolTree
        smiles = self.data[idx]
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()
        return mol_tree

class DGLMoleculeDataset(MoleculeDataset):
    def __getitem__(self, idx):
        from .mol_tree_nx import DGLMolTree
        smiles = self.data[idx]
        mol_tree = DGLMolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()
        return mol_tree

class PropDataset(Dataset):

    def __init__(self, data_file, prop_file):
        self.prop_data = np.loadtxt(prop_file)
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        from .mol_tree import MolTree
        smiles = self.data[idx]
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()
        return mol_tree, self.prop_data[idx]

class DGLPropDataset(PropDataset):
    def __getitem__(self, idx):
        from .mol_tree_nx import DGLMolTree
        smiles = self.data[idx]
        mol_tree = DGLMolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()
        return mol_tree, self.prop_data[idx]
