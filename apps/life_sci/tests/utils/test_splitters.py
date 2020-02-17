import torch

from dgllife.utils.splitters import *
from rdkit import Chem

class TestDataset(object):
    def __init__(self):
        self.smiles = [
            'CCO',
            'C1CCCCC1',
            'O1CCOCC1',
            'C1CCCC2C1CCCC2',
            'N#N'
        ]
        self.mols = [Chem.MolFromSmiles(s) for s in self.smiles]
        self.labels = torch.arange(2 * len(self.smiles)).reshape(len(self.smiles), -1)

    def __getitem__(self, item):
        return self.smiles[item], self.mols[item]

    def __len__(self):
        return len(self.smiles)

def test_consecutive_splitter(dataset):
    ConsecutiveSplitter.train_val_test_split(dataset)
    ConsecutiveSplitter.k_fold_split(dataset)

def test_random_splitter(dataset):
    RandomSplitter.train_val_test_split(dataset, random_state=0)
    RandomSplitter.k_fold_split(dataset)

def test_molecular_weight_splitter(dataset):
    MolecularWeightSplitter.train_val_test_split(dataset)
    MolecularWeightSplitter.k_fold_split(dataset, mols=dataset.mols)

def test_scaffold_splitter(dataset):
    ScaffoldSplitter.train_val_test_split(dataset, include_chirality=True)
    ScaffoldSplitter.k_fold_split(dataset, mols=dataset.mols)

def test_single_task_stratified_splitter(dataset):
    SingleTaskStratifiedSplitter.train_val_test_split(dataset, dataset.labels, 1)
    SingleTaskStratifiedSplitter.k_fold_split(dataset, dataset.labels, 1)

if __name__ == '__main__':
    dataset = TestDataset()
    test_consecutive_splitter(dataset)
    test_random_splitter(dataset)
    test_molecular_weight_splitter(dataset)
    test_scaffold_splitter(dataset)
    test_single_task_stratified_splitter(dataset)
