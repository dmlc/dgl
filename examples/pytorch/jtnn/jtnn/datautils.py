from torch.utils.data import Dataset
import numpy as np

import dgl
from dgl.data.utils import download, extract_archive, get_download_dir

_url = 'https://www.dropbox.com/s/4ypr0e0abcbsvoh/jtnn.zip?dl=1'

class JTNNDataset(Dataset):
    def __init__(self, data, vocab):
        self.dir = get_download_dir()
        self.zip_file_path='{}/jtnn.zip'.format(self.dir)
        download(_url, path=self.zip_file_path)
        extract_archive(self.zip_file_path, '{}/jtnn'.format(self.dir))
        print('Loading data...')
        data_file = '{}/jtnn/{}.txt'.format(self.dir, data)
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]
        self.vocab_file = '{}/jtnn/{}.txt'.format(self.dir, vocab)
        print('Loading finished.')
        print('\tNum samples:', len(self.data))
        print('\tVocab file:', self.vocab_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from .mol_tree_nx import DGLMolTree
        smiles = self.data[idx]
        mol_tree = DGLMolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()
        return mol_tree
