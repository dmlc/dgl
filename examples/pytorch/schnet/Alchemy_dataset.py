# -*- coding:utf-8 -*-
"""Example dataloader of Tencent Alchemy Dataset
https://alchemy.tencent.com/
"""
import os
import zipfile
import os.path as osp
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import dgl
from dgl.data.utils import download
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pathlib
import pandas as pd
import numpy as np
_urls = {'Alchemy': 'https://alchemy.tencent.com/data/'}


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
            type = atom.GetAtomicNum()
            aromatic = atom.GetIsAromatic()
            hybridization = atom.GetHybridization()
            num_h = atom.GetTotalNumHs()
            atom_feats_dict['pos'].append(torch.FloatTensor(geom[u]))
            atom_feats_dict['node_type'].append(type)

            h_u = []
            h_u += [
                int(symbol == x) for x in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']
            ]
            h_u.append(type)
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

    def sdf_to_dgl(self, sdf_file, self_loop=False):
        """
        Read sdf file and convert to dgl_graph
        Args:
            sdf_file: path of sdf file
            self_loop: Whetaher to add self loop
        Returns:
            g: DGLGraph
            l: related labels
        """
        sdf = open(str(sdf_file)).read()
        mol = Chem.MolFromMolBlock(sdf, removeHs=False)

        g = dgl.DGLGraph()

        # add nodes
        num_atoms = mol.GetNumAtoms()
        atom_feats = self.alchemy_nodes(mol)
        g.add_nodes(num=num_atoms, data=atom_feats)

        # add edges
        # The model we were interested assumes a complete graph.
        # If this is not the case, do the code below instead
        #
        # for bond in mol.GetBonds():
        #     u = bond.GetBeginAtomIdx()
        #     v = bond.GetEndAtomIdx()
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

        # for val/test set, labels are molecule ID
        l = torch.FloatTensor(self.target.loc[int(sdf_file.stem)].tolist()) \
            if self.mode == 'dev' else torch.LongTensor([int(sdf_file.stem)])
        return (g, l)

    def __init__(self, mode='dev', transform=None):
        assert mode in ['dev', 'valid',
                        'test'], "mode should be dev/valid/test"
        self.mode = mode
        self.transform = transform
        self.file_dir = pathlib.Path('./Alchemy_data', mode)
        self.zip_file_path = pathlib.Path('./Alchemy_data', '%s.zip' % mode)
        download(_urls['Alchemy'] + "%s.zip" % mode,
                 path=str(self.zip_file_path))
        if not os.path.exists(str(self.file_dir)):
            archive = zipfile.ZipFile(self.zip_file_path)
            archive.extractall('./Alchemy_data')
            archive.close()

        self._load()

    def _load(self):
        if self.mode == 'dev':
            target_file = pathlib.Path(self.file_dir, "train.csv")
            self.target = pd.read_csv(target_file,
                                      index_col=0,
                                      usecols=[
                                          'gdb_idx',
                                      ] +
                                      ['property_%d' % x for x in range(12)])
            self.target = self.target[['property_%d' % x for x in range(12)]]

        sdf_dir = pathlib.Path(self.file_dir, "sdf")
        self.graphs, self.labels = [], []
        for sdf_file in sdf_dir.glob("**/*.sdf"):
            result = self.sdf_to_dgl(sdf_file)
            if result is None:
                continue
            self.graphs.append(result[0])
            self.labels.append(result[1])
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


if __name__ == '__main__':
    alchemy_dataset = TencentAlchemyDataset()
    device = torch.device('cpu')
    # To speed up the training with multi-process data loader,
    # the num_workers could be set to > 1 to
    alchemy_loader = DataLoader(dataset=alchemy_dataset,
                                batch_size=20,
                                collate_fn=batcher(),
                                shuffle=False,
                                num_workers=0)

    for step, batch in enumerate(alchemy_loader):
        print("bs =", batch.graph.batch_size)
        print('feature size =', batch.graph.ndata['n_feat'].size())
        print('pos size =', batch.graph.ndata['pos'].size())
        print('edge feature size =', batch.graph.edata['e_feat'].size())
        print('edge distance size =', batch.graph.edata['distance'].size())
        print('label size=', batch.label.size())
        print(dgl.sum_nodes(batch.graph, 'n_feat').size())
        break
