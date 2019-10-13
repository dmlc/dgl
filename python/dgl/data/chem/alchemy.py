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
from ..utils import download, get_download_dir, _get_dgl_url, save_graphs, load_graphs
from ... import backend as F

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
        symbol = atom.GetSymbol()
        atom_type = atom.GetAtomicNum()
        aromatic = atom.GetIsAromatic()
        hybridization = atom.GetHybridization()
        num_h = atom.GetTotalNumHs()
        atom_feats_dict['node_type'].append(atom_type)

        h_u = []
        h_u += [int(symbol == x) for x in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']]
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
        processed one for faster speed. Default to be False.
    """
    def __init__(self, mode='dev', from_raw=False):
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

        self._load()

    def _load(self):
        if not self.from_raw:
            self.graphs, label_dict = load_graphs(osp.join(self.file_dir, "%s_graphs.bin" % self.mode))
            self.labels = label_dict['labels']
            with open(osp.join(self.file_dir, "%s_smiles.pkl" % self.mode), 'rb') as f:
                self.smiles = pickle.load(f)
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
                graph = mol_to_complete_graph(mol, atom_featurizer=alchemy_nodes,
                                              bond_featurizer=alchemy_edges)
                smiles = Chem.MolToSmiles(mol)
                self.smiles.append(smiles)
                self.graphs.append(graph)
                label = F.tensor(np.array(label[1].tolist()).astype(np.float32))
                self.labels.append(label)

            save_graphs(osp.join(self.file_dir, "%s_graphs.bin" % self.mode), self.graphs,
                        labels={'labels': F.stack(self.labels, dim=0)})
            with open(osp.join(self.file_dir, "%s_smiles.pkl" % self.mode), 'wb') as f:
                pickle.dump(self.smiles, f)

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
