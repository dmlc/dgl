import hashlib
import os
import pickle

import pandas as pd
from ogb.utils import smiles2graph
from tqdm import tqdm

from .. import backend as F

from ..convert import graph as dgl_graph
from .dgl_dataset import DGLDataset
from .utils import download, load_graphs, save_graphs, Subset


class PeptidesStructuralDataset(DGLDataset):
    r"""Peptides structure dataset for the graph classification task.

    DGL dataset of 15,535 small peptides represented as their molecular
    graph (SMILES) with 11 regression targets derived from the peptide's
    3D structure.

    The original amino acid sequence representation is provided in
    'peptide_seq' and the distance between atoms in 'self_dist_matrix' field
    of the dataset file, but not used here as any part of the input.

    The 11 regression targets were precomputed from molecule XYZ:
        Inertia_mass_[a-c]: The principal component of the inertia of the
            mass, with some normalizations. Sorted
        Inertia_valence_[a-c]: The principal component of the inertia of the
            Hydrogen atoms. This is basically a measure of the 3D
            distribution of hydrogens. Sorted
        length_[a-c]: The length around the 3 main geometric axis of
            the 3D objects (without considering atom types). Sorted
        Spherocity: SpherocityIndex descriptor computed by
            rdkit.Chem.rdMolDescriptors.CalcSpherocityIndex
        Plane_best_fit: Plane of best fit (PBF) descriptor computed by
            rdkit.Chem.rdMolDescriptors.CalcPBF

    Statistics:
    Train examples: 10,873
    Valid examples: 2,331
    Test examples: 2,331
    Average number of nodes: 150.94
    Average number of edges: 307.30
    Number of atom types: 9
    Number of bond types: 3

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: "~/.dgl/".
    force_reload : bool
        Whether to reload the dataset.
        Default: False.
    verbose : bool
        Whether to print out progress information.
        Default: False.
    smiles2graph (callable):
        A callable function that converts a SMILES string into a graph object.
        * The default smiles2graph requires rdkit to be installed *

    Examples
    ---------
    >>> from dgl.data import PeptidesStructuralDataset

    >>> dataset = PeptidesStructuralDataset()
    >>> len(dataset)
    15535
    >>> dataset.num_atom_types
    9
    >>> graph, label = dataset[0]
    >>> graph
    Graph(num_nodes=119, num_edges=244,
        ndata_schemes={'feat': Scheme(shape=(9,), dtype=torch.int64)}
        edata_schemes={'feat': Scheme(shape=(3,), dtype=torch.int64)})

    >>> split_dict = dataset.get_idx_split()
    >>> trainset = dataset[split_dict["train"]]
    >>> graph, label = trainset[0]
    >>> graph
    Graph(num_nodes=338, num_edges=682,
        ndata_schemes={'feat': Scheme(shape=(9,), dtype=torch.int64)}
        edata_schemes={'feat': Scheme(shape=(3,), dtype=torch.int64)})
    """

    def __init__(
        self,
        raw_dir=None,
        force_reload=None,
        verbose=None,
        smiles2graph=smiles2graph,
    ):
        self.smiles2graph = smiles2graph
        # MD5 hash of the dataset file.
        self.version = "9786061a34298a0684150f2e4ff13f47"
        self.url_stratified_split = """
        https://www.dropbox.com/s/9dfifzft1hqgow6/splits_random_stratified_peptide_structure.pickle?dl=1
        """
        self.md5sum_stratified_split = "5a0114bdadc80b94fc7ae974f13ef061"

        super(PeptidesStructuralDataset, self).__init__(
            name="Peptides-struc",
            raw_dir=raw_dir,
            url="""
            https://www.dropbox.com/s/464u3303eu2u4zp/peptide_structure_dataset.csv.gz?dl=1
            """,
            force_reload=force_reload,
            verbose=verbose,
        )

    @property
    def raw_data_path(self):
        return os.path.join(self.raw_path, "peptide_structure_dataset.csv.gz")

    @property
    def split_data_path(self):
        return os.path.join(
            self.raw_path, "splits_random_stratified_peptide_structure.pickle"
        )

    @property
    def graph_path(self):
        return os.path.join(self.save_path, "Peptides-struc.bin")

    @property
    def num_atom_types(self):
        return 9

    @property
    def num_bond_types(self):
        return 3

    def _md5sum(self, path):
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()

    def download(self):
        path = download(self.url, path=self.raw_data_path)
        # Save to disk the MD5 hash of the downloaded file.
        hash = self._md5sum(path)
        if hash != self.version:
            raise ValueError("Unexpected MD5 hash of the downloaded file")
        open(os.path.join(self.raw_path, hash), "w").close()
        # Download train/val/test splits.
        path_split = download(
            self.url_stratified_split, path=self.split_data_path
        )
        assert self._md5sum(path_split) == self.md5sum_stratified_split

    def process(self):
        data_df = pd.read_csv(self.raw_data_path)
        smiles_list = data_df["smiles"]
        target_names = [
            "Inertia_mass_a",
            "Inertia_mass_b",
            "Inertia_mass_c",
            "Inertia_valence_a",
            "Inertia_valence_b",
            "Inertia_valence_c",
            "length_a",
            "length_b",
            "length_c",
            "Spherocity",
            "Plane_best_fit",
        ]
        # Normalize to zero mean and unit standard deviation.
        data_df.loc[:, target_names] = data_df.loc[:, target_names].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0
        )
        if self.verbose:
            print("Converting SMILES strings into graphs...")
        self.graphs = []
        self.labels = []
        for i in tqdm(range(len(smiles_list))):
            smiles = smiles_list[i]
            y = data_df.iloc[i][target_names]
            graph = self.smiles2graph(smiles)

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]
            DGLgraph = dgl_graph(
                (graph["edge_index"][0], graph["edge_index"][1]),
                num_nodes=graph["num_nodes"],
            )
            DGLgraph.edata["feat"] = F.zerocopy_from_numpy(
                graph["edge_feat"]
            ).to(F.int64)
            DGLgraph.ndata["feat"] = F.zerocopy_from_numpy(
                graph["node_feat"]
            ).to(F.int64)

            self.graphs.append(DGLgraph)
            self.labels.append(y)

        self.labels = F.tensor(self.labels, dtype=F.float32)

    def load(self):
        self.graphs, label_dict = load_graphs(self.graph_path)
        self.labels = label_dict["labels"]

    def save(self):
        save_graphs(
            self.graph_path, self.graphs, labels={"labels": self.labels}
        )

    def has_cache(self):
        return os.path.exists(self.graph_path)

    def get_idx_split(self):
        """Get dataset splits.

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        with open(self.split_data_path, "rb") as f:
            split_dict = pickle.load(f)
        for key in split_dict.keys():
            split_dict[key] = F.zerocopy_from_numpy(split_dict[key])
        return split_dict

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get datapoint with index"""
        if F.is_tensor(idx) and idx.dim() == 1:
            return Subset(self, idx.cpu())

        if self._transform is None:
            return self.graphs[idx], self.labels[idx]
        else:
            return self._transform(self.graphs[idx]), self.labels[idx]
