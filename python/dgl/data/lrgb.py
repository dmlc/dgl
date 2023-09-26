import hashlib
import os
import pickle

import pandas as pd
from ogb.utils import smiles2graph
from tqdm import tqdm

from .. import backend as F

from ..convert import graph as dgl_graph
from .dgl_dataset import DGLDataset
from .utils import (
    download,
    extract_archive,
    load_graphs,
    makedirs,
    save_graphs,
    Subset,
)


class PeptidesStructuralDataset(DGLDataset):
    r"""Peptides structure dataset for the graph regression task.

    DGL dataset of 15,535 small peptides represented as their molecular
    graph (SMILES) with 11 regression targets derived from the peptide's
    3D structure.

    The 11 regression targets were precomputed from molecules' 3D structure:
        Inertia_mass_[a-c]: The principal component of the inertia of the
            mass, with some normalizations. (Sorted)
        Inertia_valence_[a-c]: The principal component of the inertia of the
            Hydrogen atoms. This is basically a measure of the 3D
            distribution of hydrogens. (Sorted)
        length_[a-c]: The length around the 3 main geometric axis of
            the 3D objects (without considering atom types). (Sorted)
        Spherocity: SpherocityIndex descriptor computed by
            rdkit.Chem.rdMolDescriptors.CalcSpherocityIndex
        Plane_best_fit: Plane of best fit (PBF) descriptor computed by
            rdkit.Chem.rdMolDescriptors.CalcPBF

    Reference `<https://arxiv.org/abs/2206.08164.pdf>`_

    Statistics:

    - Train examples: 10,873
    - Valid examples: 2,331
    - Test examples: 2,331
    - Average number of nodes: 150.94
    - Average number of edges: 307.30
    - Number of atom types: 9
    - Number of bond types: 3

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
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.
    smiles2graph : callable
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
        transform=None,
        smiles2graph=smiles2graph,
    ):
        self.smiles2graph = smiles2graph
        # MD5 hash of the dataset file.
        self.md5sum_data = "9786061a34298a0684150f2e4ff13f47"
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
            transform=transform,
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
        if hash != self.md5sum_data:
            raise ValueError("Unexpected MD5 hash of the downloaded file")
        open(os.path.join(self.raw_path, hash), "w").close()
        # Download train/val/test splits.
        path_split = download(
            self.url_stratified_split, path=self.split_data_path
        )
        hash_split = self._md5sum(path_split)
        if hash_split != self.md5sum_stratified_split:
            raise ValueError("Unexpected MD5 hash of the split file")

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


class PeptidesFunctionalDataset(DGLDataset):
    r"""Peptides functional dataset for the graph classification task.

    DGL dataset of 15,535 peptides represented as their molecular graph
    (SMILES) with 10-way multi-task binary classification of their
    functional classes.

    The 10 classes represent the following functional classes (in order):
        ['antifungal', 'cell_cell_communication', 'anticancer',
        'drug_delivery_vehicle', 'antimicrobial', 'antiviral',
        'antihypertensive', 'antibacterial', 'antiparasitic', 'toxic']

    Reference `<https://arxiv.org/abs/2206.08164.pdf>`_

    Statistics:

    - Train examples: 10,873
    - Valid examples: 2,331
    - Test examples: 2,331
    - Average number of nodes: 150.94
    - Average number of edges: 307.30
    - Number of atom types: 9
    - Number of bond types: 3

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
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.
    smiles2graph (callable):
        A callable function that converts a SMILES string into a graph object.
        * The default smiles2graph requires rdkit to be installed *

    Examples
    ---------
    >>> from dgl.data import PeptidesFunctionalDataset

    >>> dataset = PeptidesFunctionalDataset()
    >>> len(dataset)
    15535
    >>> dataset.num_classes
    10
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
        transform=None,
        smiles2graph=smiles2graph,
    ):
        self.smiles2graph = smiles2graph
        # MD5 hash of the dataset file.
        self.md5sum_data = "701eb743e899f4d793f0e13c8fa5a1b4"
        self.url_stratified_split = """
        https://www.dropbox.com/s/j4zcnx2eipuo0xz/splits_random_stratified_peptide.pickle?dl=1
        """
        self.md5sum_stratified_split = "5a0114bdadc80b94fc7ae974f13ef061"

        super(PeptidesFunctionalDataset, self).__init__(
            name="Peptides-func",
            raw_dir=raw_dir,
            url="""
            https://www.dropbox.com/s/ol2v01usvaxbsr8/peptide_multi_class_dataset.csv.gz?dl=1
            """,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    @property
    def raw_data_path(self):
        return os.path.join(self.raw_path, "peptide_multi_class_dataset.csv.gz")

    @property
    def split_data_path(self):
        return os.path.join(
            self.raw_path, "splits_random_stratified_peptide.pickle"
        )

    @property
    def graph_path(self):
        return os.path.join(self.save_path, "Peptides-func.bin")

    @property
    def num_atom_types(self):
        return 9

    @property
    def num_bond_types(self):
        return 3

    @property
    def num_classes(self):
        return 10

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
        if hash != self.md5sum_data:
            raise ValueError("Unexpected MD5 hash of the downloaded file")
        open(os.path.join(self.raw_path, hash), "w").close()
        # Download train/val/test splits.
        path_split = download(
            self.url_stratified_split, path=self.split_data_path
        )
        hash_split = self._md5sum(path_split)
        if hash_split != self.md5sum_stratified_split:
            raise ValueError("Unexpected MD5 hash of the split file")

    def process(self):
        data_df = pd.read_csv(self.raw_data_path)
        smiles_list = data_df["smiles"]
        if self.verbose:
            print("Converting SMILES strings into graphs...")
        self.graphs = []
        self.labels = []
        for i in tqdm(range(len(smiles_list))):
            smiles = smiles_list[i]
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
            self.labels.append(eval(data_df["labels"].iloc[i]))
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


class VOCSuperpixelsDataset(DGLDataset):
    r"""VOCSuperpixels dataset for the node classification task.

    DGL dataset of Pascal VOC Superpixels which contains image superpixels
    and a semantic segmentation label for each node superpixel.

    color map
    0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle,
    6=bus, 7=car, 8=cat, 9=chair, 10=cow,
    11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person,
    16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

    Reference `<https://arxiv.org/abs/2206.08164.pdf>`_

    Statistics:

    - Train examples: 8,498
    - Valid examples: 1,428
    - Test examples: 1,429
    - Average number of nodes: 479.40
    - Average number of edges: 2,710.48

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: "~/.dgl/".
    split : str
        Should be chosen from ["train", "val", "test"]
        Default: "train".
    construct_format : str, optional
        Option to select the graph construction format.
        Should be chosen from ["edge_wt_only_coord", "edge_wt_coord_feat", "edge_wt_region_boundary"]
        "edge_wt_only_coord": the graphs are 8-nn graphs with the edge weights
        computed based on only spatial coordinates of superpixel nodes.
        "edge_wt_coord_feat": the graphs are 8-nn graphs with the edge weights
        computed based on combination of spatial coordinates and feature
        values of superpixel nodes.
        "edge_wt_region_boundary": the graphs region boundary graphs where two
        regions (i.e. superpixel nodes) have an edge between them if they share
        a boundary in the original image.
        Default: "edge_wt_region_boundary".
    slic_compactness : int, optional
        Option to select compactness of slic that was used for superpixels
        Should be chosen from [10, 30]
        Default: 30.
    force_reload : bool
        Whether to reload the dataset.
        Default: False.
    verbose : bool
        Whether to print out progress information.
        Default: False.
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.

    Examples
    ---------
    >>> from dgl.data import VOCSuperpixelsDataset

    >>> train_dataset = VOCSuperpixelsDataset(split="train")
    >>> len(train_dataset)
    8498
    >>> train_dataset.num_classes
    21
    >>> graph = train_dataset[0]
    >>> graph
    Graph(num_nodes=460, num_edges=2632,
        ndata_schemes={'feat': Scheme(shape=(14,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int32)}
        edata_schemes={'feat': Scheme(shape=(2,), dtype=torch.float32)})
    """

    urls = {
        10: {
            "edge_wt_only_coord": """
            https://www.dropbox.com/s/rk6pfnuh7tq3t37/voc_superpixels_edge_wt_only_coord.zip?dl=1
            """,
            "edge_wt_coord_feat": """
            https://www.dropbox.com/s/2a53nmfp6llqg8y/voc_superpixels_edge_wt_coord_feat.zip?dl=1
            """,
            "edge_wt_region_boundary": """
            https://www.dropbox.com/s/6pfz2mccfbkj7r3/voc_superpixels_edge_wt_region_boundary.zip?dl=1
            """,
        },
        30: {
            "edge_wt_only_coord": """
            https://www.dropbox.com/s/toqulkdpb1jrswk/voc_superpixels_edge_wt_only_coord.zip?dl=1
            """,
            "edge_wt_coord_feat": """
            https://www.dropbox.com/s/xywki8ysj63584d/voc_superpixels_edge_wt_coord_feat.zip?dl=1
            """,
            "edge_wt_region_boundary": """
            https://www.dropbox.com/s/8x722ai272wqwl4/voc_superpixels_edge_wt_region_boundary.zip?dl=1
            """,
        },
    }

    def __init__(
        self,
        raw_dir=None,
        split="train",
        construct_format="edge_wt_region_boundary",
        slic_compactness=30,
        force_reload=None,
        verbose=None,
        transform=None,
    ):
        self.construct_format = construct_format
        self.slic_compactness = slic_compactness
        assert split in ["train", "val", "test"]
        assert construct_format in [
            "edge_wt_only_coord",
            "edge_wt_coord_feat",
            "edge_wt_region_boundary",
        ]
        assert slic_compactness in [10, 30]
        self.split = split
        super(VOCSuperpixelsDataset, self).__init__(
            name="PascalVOC-SP",
            raw_dir=raw_dir,
            url=self.urls[self.slic_compactness][self.construct_format],
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    @property
    def save_path(self):
        return os.path.join(
            self.raw_path,
            "slic_compactness_" + str(self.slic_compactness),
            self.construct_format,
        )

    @property
    def raw_data_path(self):
        return os.path.join(self.save_path, f"{self.split}.pickle")

    @property
    def graph_path(self):
        return os.path.join(self.save_path, f"processed_{self.split}.pkl")

    @property
    def num_classes(self):
        r"""Number of classes for each node."""
        return 21

    def __len__(self):
        r"""The number of examples in the dataset."""
        return len(self.graphs)

    def download(self):
        zip_file_path = os.path.join(
            self.raw_path, "voc_superpixels_" + self.construct_format + ".zip"
        )
        path = download(self.url, path=zip_file_path)
        extract_archive(path, self.raw_path, overwrite=True)
        makedirs(self.save_path)
        os.rename(
            os.path.join(
                self.raw_path, "voc_superpixels_" + self.construct_format
            ),
            self.save_path,
        )
        os.unlink(path)

    def process(self):
        with open(self.raw_data_path, "rb") as f:
            graphs = pickle.load(f)

        self.graphs = []
        for idx in tqdm(
            range(len(graphs)), desc=f"Processing {self.split} dataset"
        ):
            graph = graphs[idx]

            """
            Each `graph` is a tuple (x, edge_attr, edge_index, y)
                Shape of x : [num_nodes, 14]
                Shape of edge_attr : [num_edges, 1] or [num_edges, 2]
                Shape of edge_index : [2, num_edges]
                Shape of y : [num_nodes]
            """
            DGLgraph = dgl_graph(
                (graph[2][0], graph[2][1]),
                num_nodes=len(graph[3]),
            )
            DGLgraph.ndata["feat"] = graph[0].to(F.float32)
            DGLgraph.edata["feat"] = graph[1].to(F.float32)
            DGLgraph.ndata["label"] = F.tensor(graph[3])
            self.graphs.append(DGLgraph)

    def load(self):
        with open(self.graph_path, "rb") as f:
            f = pickle.load(f)
            self.graphs = f

    def save(self):
        with open(os.path.join(self.graph_path), "wb") as f:
            pickle.dump(self.graphs, f)

    def has_cache(self):
        return os.path.exists(self.graph_path)

    def __getitem__(self, idx):
        r"""Get the idx^th sample.

        Parameters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        :class:`dgl.DGLGraph`
            graph structure, node features, node labels and edge features.

            - ``ndata['feat']``: node features
            - ``ndata['label']``: node labels
            - ``edata['feat']``: edge features
        """
        if F.is_tensor(idx) and idx.dim() == 1:
            return Subset(self, idx.cpu())

        if self._transform is None:
            return self.graphs[idx]
        else:
            return self._transform(self.graphs[idx])
