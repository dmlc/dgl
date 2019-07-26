# -*- coding:utf-8 -*-
"""Example dataloader of Tencent Alchemy Dataset 
https://alchemy.tencent.com/
"""
import os.path as osp
import networkx as nx
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import dgl
from dgl.data.utils import get_download_dir
from dgl.data.utils import download
from dgl.data.utils import extract_archive
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import namedtuple
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

    def alchemy_nodes(self, g):
        feat = []
        for n, d in g.nodes(data=True):
            h_t = []
            # Atom type (One-hot H, C, N, O F)
            h_t += [
                int(d['node_str'] == x)
                for x in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']
            ]
            # Atomic number
            h_t.append(d['node_type'])
            # Acceptor
            h_t.append(d['acceptor'])
            # Donor
            h_t.append(d['donor'])
            # Aromatic
            h_t.append(int(d['aromatic']))
            # Hybradization
            h_t += [
                int(d['hybridization'] == x)
                for x in (Chem.rdchem.HybridizationType.SP,
                          Chem.rdchem.HybridizationType.SP2,
                          Chem.rdchem.HybridizationType.SP3)
            ]
            h_t.append(d['num_h'])
            feat.append((n, torch.FloatTensor(h_t)))

        nx.set_node_attributes(g, dict(feat), "n_feat")

    def alchemy_edges(self, g):
        e = {}
        for n1, n2, d in g.edges(data=True):
            e_t = [
                float(d['b_type'] == x)
                for x in (Chem.rdchem.BondType.SINGLE,
                          Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE,
                          Chem.rdchem.BondType.AROMATIC, "NoBond")
            ]
            e[(n1, n2)] = e_t
        nx.set_edge_attributes(g, e, "e_feat")

    # sdf file reader for Alchemy dataset
    def sdf_graph_reader(self, sdf_file):

        with open(str(sdf_file), 'r') as f:
            sdf_string = f.read()
        mol = Chem.MolFromMolBlock(sdf_string, removeHs=False)
        if mol is None:
            print("rdkit can not parsing", sdf_file)
            return None
        feats = self.chem_feature_factory.GetFeaturesForMol(mol)

        g = nx.DiGraph()
        l = torch.FloatTensor(self.target.loc[int(sdf_file.stem)].tolist()) \
            if self.mode == 'dev' else torch.LongTensor([int(sdf_file.stem)])

        # Create nodes
        assert len(mol.GetConformers()) == 1
        geom = mol.GetConformers()[0].GetPositions()
        for i in range(mol.GetNumAtoms()):
            atom_i = mol.GetAtomWithIdx(i)
            g.add_node(i,
                       node_str=atom_i.GetSymbol(),
                       node_type=atom_i.GetAtomicNum(),
                       acceptor=0,
                       donor=0,
                       aromatic=atom_i.GetIsAromatic(),
                       hybridization=atom_i.GetHybridization(),
                       num_h=atom_i.GetTotalNumHs(),
                       pos=torch.FloatTensor(geom[i]))

        for i in range(len(feats)):
            if feats[i].GetFamily() == 'Donor':
                node_list = feats[i].GetAtomIds()
                for i in node_list:
                    g.node[i]['donor'] = 1
            elif feats[i].GetFamily() == 'Acceptor':
                node_list = feats[i].GetAtomIds()
                for i in node_list:
                    g.node[i]['acceptor'] = 1
        # Read Edges
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                # remove self-loop
                if i == j:
                    continue
                if e_ij is not None:
                    g.add_edge(i, j, b_type=e_ij.GetBondType())
                else:
                    g.add_edge(i, j, b_type="NoBond")
                g.add_edge(i, j, distance=np.linalg.norm(geom[i] - geom[j]))

        self.alchemy_nodes(g)
        self.alchemy_edges(g)
        ret = dgl.DGLGraph()
        ret.from_networkx(g,
                          node_attrs=['n_feat', 'pos', 'node_type'],
                          edge_attrs=['e_feat', 'distance'])
        ret.edata["distance"] = ret.edata["distance"].reshape(-1, 1)

        return ret, l

    def __init__(self, mode='dev', transform=None):
        assert mode in ['dev', 'valid',
                        'test'], "mode should be dev/valid/test"
        self.mode = mode
        self.transform = transform
        self.file_dir = pathlib.Path(get_download_dir(), mode)
        self.zip_file_path = pathlib.Path(get_download_dir(), '%s.zip' % mode)
        download(_urls['Alchemy'] + "%s.zip" % mode,
                 path=str(self.zip_file_path))
        extract_archive(str(self.zip_file_path), str(self.file_dir))

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
        cnt = 0
        for sdf_file in sdf_dir.glob("**/*.sdf"):
            result = self.sdf_graph_reader(sdf_file)
            if result is None:
                continue
            cnt += 1
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
    alchemy_loader = DataLoader(dataset=alchemy_dataset,
                                batch_size=20,
                                collate_fn=batcher(device),
                                shuffle=False,
                                num_workers=0)

    for step, batch in enumerate(alchemy_loader):
        print(next(iter(batch.graph.ndata.values())).device)
        print("bs =", batch.graph.batch_size)
        print('feature size =', batch.graph.ndata['n_feat'].size())
        print('pos size =', batch.graph.ndata['pos'].size())
        print('edge feature size =', batch.graph.edata['e_feat'].size())
        print('edge distance size =', batch.graph.edata['distance'].size())
        print('label size=', batch.label.size())
        print(dgl.sum_nodes(batch.graph, 'n_feat').size())
        break
