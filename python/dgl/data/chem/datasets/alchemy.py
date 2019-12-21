# -*- coding:utf-8 -*-
"""Example dataloader of Tencent Alchemy Dataset
https://alchemy.tencent.com/
"""
import numpy as np
import os
import os.path as osp
import pathlib
import zipfile
from collections import defaultdict

from ..utils import mol_to_complete_graph, atom_type_one_hot, \
    atom_hybridization_one_hot, atom_is_aromatic
from ...utils import download, get_download_dir, _get_dgl_url, save_graphs, load_graphs
from .... import backend as F

try:
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import ChemicalFeatures
    from rdkit import RDConfig
except ImportError:
    pass

def alchemy_nodes(mol):
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
        atom_type = atom.GetAtomicNum()
        num_h = atom.GetTotalNumHs()
        atom_feats_dict['node_type'].append(atom_type)

        h_u = []
        h_u += atom_type_one_hot(atom, ['H', 'C', 'N', 'O', 'F', 'S', 'Cl'])
        h_u.append(atom_type)
        h_u.append(is_acceptor[u])
        h_u.append(is_donor[u])
        h_u += atom_is_aromatic(atom)
        h_u += atom_hybridization_one_hot(atom, [Chem.rdchem.HybridizationType.SP,
                                                 Chem.rdchem.HybridizationType.SP2,
                                                 Chem.rdchem.HybridizationType.SP3])
        h_u.append(num_h)
        atom_feats_dict['n_feat'].append(F.tensor(np.array(h_u).astype(np.float32)))

    atom_feats_dict['n_feat'] = F.stack(atom_feats_dict['n_feat'], dim=0)
    atom_feats_dict['node_type'] = F.tensor(np.array(
        atom_feats_dict['node_type']).astype(np.int64))

    return atom_feats_dict

def alchemy_edges(mol, self_loop=False):
    """Featurization for all bonds in a molecule.
    The bond indices will be preserved.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object
    self_loop : bool
        Whether to add self loops. Default to be False.

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

class TencentAlchemyDataset(object):
    """
    Developed by the Tencent Quantum Lab, the dataset lists 12 quantum mechanical
    properties of 130, 000+ organic molecules, comprising up to 12 heavy atoms
    (C, N, O, S, F and Cl), sampled from the GDBMedChem database. These properties
    have been calculated using the open-source computational chemistry program
    Python-based Simulation of Chemistry Framework (PySCF).

    For more details, check the `paper <https://arxiv.org/abs/1906.09427>`__.

    Parameters
    ----------
    mode : str
        'dev', 'valid' or 'test', separately for training, validation and test.
        Default to be 'dev'. Note that 'test' is not available as the Alchemy
        contest is ongoing.
    from_raw : bool
        Whether to process the dataset from scratch or use a
        processed one for faster speed. If you use different ways
        to featurize atoms or bonds, you should set this to be True.
        Default to be False.
    mol_to_graph: callable, str -> DGLGraph
        A function turning an RDKit molecule instance into a DGLGraph.
        Default to :func:`dgl.data.chem.mol_to_complete_graph`.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. By default, we construct graphs where nodes represent atoms
        and node features represent atom features. We store the atomic numbers under the
        name ``"node_type"`` and store the atom features under the name ``"n_feat"``.
        The atom features include:
        * One hot encoding for atom types
        * Atomic number of atoms
        * Whether the atom is a donor
        * Whether the atom is an acceptor
        * Whether the atom is aromatic
        * One hot encoding for atom hybridization
        * Total number of Hs on the atom
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. By default, we construct edges between every pair of atoms,
        excluding the self loops. We store the distance between the end atoms under the name
        ``"distance"`` and store the edge features under the name ``"e_feat"``. The edge
        features represent one hot encoding of edge types (bond types and non-bond edges).
    """
    def __init__(self, mode='dev', from_raw=False,
                 mol_to_graph=mol_to_complete_graph,
                 node_featurizer=alchemy_nodes,
                 edge_featurizer=alchemy_edges):
        if mode == 'test':
            raise ValueError('The test mode is not supported before '
                             'the Alchemy contest finishes.')

        assert mode in ['dev', 'valid', 'test'], \
            'Expect mode to be dev, valid or test, got {}.'.format(mode)

        self.mode = mode

        # Construct DGLGraphs from raw data or use the preprocessed data
        self.from_raw = from_raw
        file_dir = osp.join(get_download_dir(), 'Alchemy_data')

        if not from_raw:
            file_name = "%s_processed_dgl" % (mode)
        else:
            file_name = "%s_single_sdf" % (mode)
        self.file_dir = pathlib.Path(file_dir, file_name)

        self._url = 'dataset/alchemy/'
        self.zip_file_path = pathlib.Path(file_dir, file_name + '.zip')
        download(_get_dgl_url(self._url + file_name + '.zip'), path=str(self.zip_file_path))
        if not os.path.exists(str(self.file_dir)):
            archive = zipfile.ZipFile(self.zip_file_path)
            archive.extractall(file_dir)
            archive.close()

        self._load(mol_to_graph, node_featurizer, edge_featurizer)

    def _load(self, mol_to_graph, node_featurizer, edge_featurizer):
        if not self.from_raw:
            self.graphs, label_dict = load_graphs(osp.join(self.file_dir, "%s_graphs.bin" % self.mode))
            self.labels = label_dict['labels']
            with open(osp.join(self.file_dir, "%s_smiles.txt" % self.mode), 'r') as f:
                smiles_ = f.readlines()
                self.smiles = [s.strip() for s in smiles_]
        else:
            print('Start preprocessing dataset...')
            target_file = pathlib.Path(self.file_dir, "%s_target.csv" % self.mode)
            self.target = pd.read_csv(
                target_file,
                index_col=0,
                usecols=['gdb_idx',] + ['property_%d' % x for x in range(12)])
            self.target = self.target[['property_%d' % x for x in range(12)]]
            self.graphs, self.labels, self.smiles = [], [], []

            supp = Chem.SDMolSupplier(osp.join(self.file_dir, self.mode + ".sdf"))
            cnt = 0
            dataset_size = len(self.target)
            for mol, label in zip(supp, self.target.iterrows()):
                cnt += 1
                print('Processing molecule {:d}/{:d}'.format(cnt, dataset_size))
                graph = mol_to_graph(mol, node_featurizer=node_featurizer,
                                     edge_featurizer=edge_featurizer)
                smiles = Chem.MolToSmiles(mol)
                self.smiles.append(smiles)
                self.graphs.append(graph)
                label = F.tensor(np.array(label[1].tolist()).astype(np.float32))
                self.labels.append(label)

            save_graphs(osp.join(self.file_dir, "%s_graphs.bin" % self.mode), self.graphs,
                        labels={'labels': F.stack(self.labels, dim=0)})
            with open(osp.join(self.file_dir, "%s_smiles.txt" % self.mode), 'w') as f:
                for s in self.smiles:
                    f.write(s + '\n')

        self.set_mean_and_std()
        print(len(self.graphs), "loaded!")

    def __getitem__(self, item):
        """Get datapoint with index

        Parameters
        ----------
        item : int
            Datapoint index

        Returns
        -------
        str
            SMILES for the ith datapoint
        DGLGraph
            DGLGraph for the ith datapoint
        Tensor of dtype float32
            Labels of the datapoint for all tasks
        """
        return self.smiles[item], self.graphs[item], self.labels[item]

    def __len__(self):
        """Length of the dataset

        Returns
        -------
        int
            Length of Dataset
        """
        return len(self.graphs)

    def set_mean_and_std(self, mean=None, std=None):
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
