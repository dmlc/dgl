# -*- coding:utf-8 -*-
"""Example dataloader of Tencent Alchemy Dataset
https://alchemy.tencent.com/
"""

import os
import os.path as osp
import zipfile
import dgl
from dgl.data.utils import download
import pickle
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pathlib
from ..utils import get_download_dir, download, _get_dgl_url
try:
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import ChemicalFeatures
    from rdkit import RDConfig
except ImportError:
    pass
_urls = {'Alchemy': 'https://alchemy.tencent.com/data/dgl/'}


class AlchemyBatcher:
    def __init__(self, graph=None, label=None):
        self.graph = graph
        self.label = label


def batcher():
    def batcher_dev(batch):
        graphs, labels = zip(*batch)
        batch_graphs = dgl.batch(graphs)
        labels = torch.stack(labels, 0)
        return AlchemyBatcher(graph=batch_graphs, label=labels)

    return batcher_dev


class TencentAlchemyDataset(Dataset):

    fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

    def alchemy_nodes(self, mol):
        """Featurization for all atoms in a molecule. The atom indices
        will be preserved.

        Args:
            mol : rdkit.Chem.rdchem.Mol
              RDKit molecule object
        Returns
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
            atom_feats_dict['pos'].append(torch.FloatTensor(geom[u]))
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
            atom_feats_dict['n_feat'].append(torch.FloatTensor(h_u))

        atom_feats_dict['n_feat'] = torch.stack(atom_feats_dict['n_feat'],
                                                dim=0)
        atom_feats_dict['pos'] = torch.stack(atom_feats_dict['pos'], dim=0)
        atom_feats_dict['node_type'] = torch.LongTensor(
            atom_feats_dict['node_type'])

        return atom_feats_dict

    def alchemy_edges(self, mol, self_loop=True):
        """Featurization for all bonds in a molecule. The bond indices
        will be preserved.

        Args:
          mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object

        Returns
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

        bond_feats_dict['e_feat'] = torch.FloatTensor(
            bond_feats_dict['e_feat'])
        bond_feats_dict['distance'] = torch.FloatTensor(
            bond_feats_dict['distance']).reshape(-1, 1)

        return bond_feats_dict

    def mol_to_dgl(self, mol, self_loop=False):
        """
        Convert RDKit molecule object to DGLGraph
        Args:
            mol: Chem.rdchem.Mol read from sdf
            self_loop: Whetaher to add self loop
        Returns:
            g: DGLGraph
            l: related labels
        """

        g = dgl.DGLGraph()

        # add nodes
        num_atoms = mol.GetNumAtoms()
        atom_feats = self.alchemy_nodes(mol)
        g.add_nodes(num=num_atoms, data=atom_feats)

        if self_loop:
            g.add_edges(
                [i for i in range(num_atoms) for j in range(num_atoms)],
                [j for i in range(num_atoms) for j in range(num_atoms)])
        else:
            g.add_edges(
                [i for i in range(num_atoms) for j in range(num_atoms - 1)], [
                    j for i in range(num_atoms)
                    for j in range(num_atoms) if i != j
                ])

        bond_feats = self.alchemy_edges(mol, self_loop)
        g.edata.update(bond_feats)

        return g

    def __init__(self, mode='dev', transform=None, from_raw=False):
        assert mode in ['dev', 'valid',
                        'test'], "mode should be dev/valid/test"
        self.mode = mode
        self.transform = transform

        # Construct the dgl graph from raw data or use the preprocessed data directly
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
                with open(osp.join(self.file_dir, "dev_graphs.pkl"),
                          "rb") as f:
                    self.graphs = pickle.load(f)
                with open(osp.join(self.file_dir, "dev_labels.pkl"),
                          "rb") as f:
                    self.labels = pickle.load(f)

            else:

                target_file = pathlib.Path(self.file_dir, "dev_target.csv")
                self.target = pd.read_csv(
                    target_file,
                    index_col=0,
                    usecols=[
                        'gdb_idx',
                    ] + ['property_%d' % x for x in range(12)])
                self.target = self.target[[
                    'property_%d' % x for x in range(12)
                ]]
                self.graphs, self.labels = [], []

                sdf_dir = pathlib.Path(self.file_dir, "sdf")
                supp = Chem.SDMolSupplier(
                    osp.join(self.file_dir, self.mode + ".sdf"))
                cnt = 0
                for sdf, label in zip(supp, self.target.iterrows()):
                    graph = self.mol_to_dgl(sdf)
                    cnt += 1
                    self.graphs.append(graph)
                    label = torch.FloatTensor(label[1].tolist())
                    self.labels.append(label)

        self.normalize()
        print(len(self.graphs), "loaded!")

    def normalize(self, mean=None, std=None):
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
        """
        Split the dataset into two AlchemySubset for train&test.
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
    """

    def __init__(self, graphs, labels, mean=0, std=1, transform=None):

        self.graphs = graphs
        self.labels = labels
        self.mean = mean
        self.std = std
        self.transform = transform
