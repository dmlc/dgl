# -*- coding:utf-8 -*-
"""Example dataloader of Tencent Alchemy Dataset
https://alchemy.tencent.com/
"""
import numpy as np
import os
import os.path as osp
import pathlib
import pickle
import zipfile
from collections import defaultdict

from .utils import mol_to_complete_graph
from ..utils import download, get_download_dir
from ...batched_graph import batch
from ... import backend as F

try:
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import ChemicalFeatures
    from rdkit import RDConfig
except ImportError:
    pass

_urls = {'Alchemy': 'https://alchemy.tencent.com/data/dgl/'}

class AlchemyBatcher(object):
    """Data structure for holding a batch of data.

    Parameters
    ----------
    graph : dgl.BatchedDGLGraph
        A batch of DGLGraphs for B molecules
    labels : tensor
        Labels for B molecules
    """
    def __init__(self, graph=None, label=None):
        self.graph = graph
        self.label = label

def batcher_dev(batch_data):
    """Batch datapoints

    Parameters
    ----------
    batch_data : list
        batch[i][0] gives the DGLGraph for the ith datapoint,
        and batch[i][1] gives the label for the ith datapoint.

    Returns
    -------
    AlchemyBatcher
        An object holding the batch of data
    """
    graphs, labels = zip(*batch_data)
    batch_graphs = batch(graphs)
    labels = F.stack(labels, 0)

    return AlchemyBatcher(graph=batch_graphs, label=labels)

class TencentAlchemyDataset(object):
    """`Tencent Alchemy Dataset <https://arxiv.org/abs/1906.09427>`__

    Parameters
    ----------
    mode : str
        'dev', 'valid' or 'test', default to be 'dev'
    transform : transform operation on DGLGraphs
        Default to be None.
    from_raw : bool
        Whether to process dataset from scratch or use a
        processed one for faster speed. Default to be False.
    """
    def __init__(self, mode='dev', transform=None, from_raw=False):
        assert mode in ['dev', 'valid', 'test'], "mode should be dev/valid/test"
        self.mode = mode
        self.transform = transform

        # Construct DGLGraphs from raw data or use the preprocessed data
        self.from_raw = from_raw
        file_dir = osp.join(get_download_dir(), './Alchemy_data')

        if not from_raw:
            file_name = "%s_processed" % (mode)
        else:
            file_name = "%s_single_sdf" % (mode)
        self.file_dir = pathlib.Path(file_dir, file_name)

        self.zip_file_path = pathlib.Path(file_dir, file_name + '.zip')
        download(_urls['Alchemy'] + file_name + '.zip',
                 path=str(self.zip_file_path))
        if not os.path.exists(str(self.file_dir)):
            archive = zipfile.ZipFile(self.zip_file_path)
            archive.extractall(file_dir)
            archive.close()

        self._load()

    def _load(self):
        if self.mode == 'dev':
            if not self.from_raw:
                with open(osp.join(self.file_dir, "dev_graphs.pkl"), "rb") as f:
                    self.graphs = pickle.load(f)
                with open(osp.join(self.file_dir, "dev_labels.pkl"), "rb") as f:
                    self.labels = pickle.load(f)
            else:
                target_file = pathlib.Path(self.file_dir, "dev_target.csv")
                self.target = pd.read_csv(
                    target_file,
                    index_col=0,
                    usecols=['gdb_idx',] + ['property_%d' % x for x in range(12)])
                self.target = self.target[['property_%d' % x for x in range(12)]]
                self.graphs, self.labels = [], []

                supp = Chem.SDMolSupplier(
                    osp.join(self.file_dir, self.mode + ".sdf"))
                cnt = 0
                for sdf, label in zip(supp, self.target.iterrows()):
                    graph = mol_to_complete_graph(sdf, atom_featurizer=self.alchemy_nodes,
                                                  bond_featurizer=self.alchemy_edges)
                    cnt += 1
                    self.graphs.append(graph)
                    label = F.tensor(np.array(label[1].tolist()).astype(np.float32))
                    self.labels.append(label)

        self.normalize()
        print(len(self.graphs), "loaded!")

    def alchemy_nodes(self, mol):
        """Featurization for all atoms in a molecule. The atom indices
        will be preserved.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object

        Returns
        -------
        atom_feats_dict : dict
            Dictionary for atom features
        """
        atom_feats_dict = defaultdict(list)
        is_donor = defaultdict(int)
        is_acceptor = defaultdict(int)

        fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        mol_feats = mol_featurizer.GetFeaturesForMol(mol)
        mol_conformers = mol.GetConformers()
        assert len(mol_conformers) == 1
        geom = mol_conformers[0].GetPositions()

        for i in range(len(mol_feats)):
            if mol_feats[i].GetFamily() == 'Donor':
                node_list = mol_feats[i].GetAtomIds()
                for u in node_list:
                    is_donor[u] = 1
            elif mol_feats[i].GetFamily() == 'Acceptor':
                node_list = mol_feats[i].GetAtomIds()
                for u in node_list:
                    is_acceptor[u] = 1

        num_atoms = mol.GetNumAtoms()
        for u in range(num_atoms):
            atom = mol.GetAtomWithIdx(u)
            symbol = atom.GetSymbol()
            atom_type = atom.GetAtomicNum()
            aromatic = atom.GetIsAromatic()
            hybridization = atom.GetHybridization()
            num_h = atom.GetTotalNumHs()
            atom_feats_dict['pos'].append(F.tensor(geom[u].astype(np.float32)))
            atom_feats_dict['node_type'].append(atom_type)

            h_u = []
            h_u += [
                int(symbol == x) for x in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']
            ]
            h_u.append(atom_type)
            h_u.append(is_acceptor[u])
            h_u.append(is_donor[u])
            h_u.append(int(aromatic))
            h_u += [
                int(hybridization == x)
                for x in (Chem.rdchem.HybridizationType.SP,
                          Chem.rdchem.HybridizationType.SP2,
                          Chem.rdchem.HybridizationType.SP3)
            ]
            h_u.append(num_h)
            atom_feats_dict['n_feat'].append(F.tensor(np.array(h_u).astype(np.float32)))

        atom_feats_dict['n_feat'] = F.stack(atom_feats_dict['n_feat'], dim=0)
        atom_feats_dict['pos'] = F.stack(atom_feats_dict['pos'], dim=0)
        atom_feats_dict['node_type'] = F.tensor(np.array(
            atom_feats_dict['node_type']).astype(np.int64))

        return atom_feats_dict

    def alchemy_edges(self, mol, self_loop=False):
        """Featurization for all bonds in a molecule.
        The bond indices will be preserved.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object

        Returns
        -------
        bond_feats_dict : dict
            Dictionary for bond features
        """
        bond_feats_dict = defaultdict(list)

        mol_conformers = mol.GetConformers()
        assert len(mol_conformers) == 1
        geom = mol_conformers[0].GetPositions()

        num_atoms = mol.GetNumAtoms()
        for u in range(num_atoms):
            for v in range(num_atoms):
                if u == v and not self_loop:
                    continue

                e_uv = mol.GetBondBetweenAtoms(u, v)
                if e_uv is None:
                    bond_type = None
                else:
                    bond_type = e_uv.GetBondType()
                bond_feats_dict['e_feat'].append([
                    float(bond_type == x)
                    for x in (Chem.rdchem.BondType.SINGLE,
                              Chem.rdchem.BondType.DOUBLE,
                              Chem.rdchem.BondType.TRIPLE,
                              Chem.rdchem.BondType.AROMATIC, None)
                ])
                bond_feats_dict['distance'].append(
                    np.linalg.norm(geom[u] - geom[v]))

        bond_feats_dict['e_feat'] = F.tensor(
            np.array(bond_feats_dict['e_feat']).astype(np.float32))
        bond_feats_dict['distance'] = F.tensor(
            np.array(bond_feats_dict['distance']).astype(np.float32)).reshape(-1 , 1)

        return bond_feats_dict

    def normalize(self, mean=None, std=None):
        """Set mean and std or compute from labels for future normalization.

        Parameters
        ----------
        mean : int or float
            Default to be None.
        std : int or float
            Default to be None.
        """
        labels = np.array([i.numpy() for i in self.labels])
        if mean is None:
            mean = np.mean(labels, axis=0)
        if std is None:
            std = np.std(labels, axis=0)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g, l = self.graphs[idx], self.labels[idx]
        if self.transform:
            g = self.transform(g)
        return g, l

    def split(self, train_size=0.8):
        """Split the dataset into two AlchemySubset for train&test.

        Parameters
        ----------
        train_size : float
            Proportion of dataset to use for training. Default to be 0.8.

        Returns
        -------
        train_set : AlchemySubset
            Dataset for training
        test_set : AlchemySubset
            Dataset for test
        """
        assert 0 < train_size < 1
        train_num = int(len(self.graphs) * train_size)
        train_set = AlchemySubset(self.graphs[:train_num],
                                  self.labels[:train_num], self.mean, self.std,
                                  self.transform)
        test_set = AlchemySubset(self.graphs[train_num:],
                                 self.labels[train_num:], self.mean, self.std,
                                 self.transform)
        return train_set, test_set

class AlchemySubset(TencentAlchemyDataset):
    """
    Sub-dataset split from TencentAlchemyDataset.
    Used to construct the training & test set.

    Parameters
    ----------
    graphs : list of DGLGraphs
        DGLGraphs for datapoints in the subset
    labels : list of tensors
        Labels for datapoints in the subset
    mean : int or float
        Mean of labels in the subset
    std : int or float
        Std of labels in the subset
    transform : transform operation on DGLGraphs
        Default to be None.
    """
    def __init__(self, graphs, labels, mean=0, std=1, transform=None):
        super(AlchemySubset, self).__init__()
        self.graphs = graphs
        self.labels = labels
        self.mean = mean
        self.std = std
        self.transform = transform
