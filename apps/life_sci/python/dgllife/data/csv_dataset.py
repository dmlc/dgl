"""Creating datasets from .csv files for molecular property prediction."""
import dgl.backend as F
import numpy as np
import os

from dgl.data.utils import save_graphs, load_graphs

__all__ = ['MoleculeCSVDataset']

class MoleculeCSVDataset(object):
    """MoleculeCSVDataset

    This is a general class for loading molecular data from :class:`pandas.DataFrame`.

    In data pre-processing, we construct a binary mask indicating the existence of labels.

    All molecules are converted into DGLGraphs. After the first-time construction, the
    DGLGraphs can be saved for reloading so that we do not need to reconstruct them every time.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe including smiles and labels. Can be loaded by pandas.read_csv(file_path).
        One column includes smiles and some other columns include labels.
    smiles_to_graph: callable, str -> DGLGraph
        A function turning a SMILES into a DGLGraph.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph.
    smiles_column: str
        Column name for smiles in ``df``.
    cache_file_path: str
        Path to store the preprocessed DGLGraphs. For example, this can be ``'dglgraph.bin'``.
    task_names : list of str or None
        Columns in the data frame corresponding to real-valued labels. If None, we assume
        all columns except the smiles_column are labels. Default to None.
    load : bool
        Whether to load the previously pre-processed dataset or pre-process from scratch.
        ``load`` should be False when we want to try different graph construction and
        featurization methods and need to preprocess from scratch. Default to True.
    log_every : bool
        Print a message every time ``log_every`` molecules are processed. Default to 1000.
    """
    def __init__(self, df, smiles_to_graph, node_featurizer, edge_featurizer,
                 smiles_column, cache_file_path, task_names=None, load=True, log_every=1000):
        self.df = df
        self.smiles = self.df[smiles_column].tolist()
        if task_names is None:
            self.task_names = self.df.columns.drop([smiles_column]).tolist()
        else:
            self.task_names = task_names
        self.n_tasks = len(self.task_names)
        self.cache_file_path = cache_file_path
        self._pre_process(smiles_to_graph, node_featurizer, edge_featurizer, load, log_every)

    def _pre_process(self, smiles_to_graph, node_featurizer, edge_featurizer, load, log_every):
        """Pre-process the dataset

        * Convert molecules from smiles format into DGLGraphs
          and featurize their atoms
        * Set missing labels to be 0 and use a binary masking
          matrix to mask them

        Parameters
        ----------
        smiles_to_graph : callable, SMILES -> DGLGraph
            Function for converting a SMILES (str) into a DGLGraph.
        node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
            Featurization for nodes like atoms in a molecule, which can be used to update
            ndata for a DGLGraph.
        edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
            Featurization for edges like bonds in a molecule, which can be used to update
            edata for a DGLGraph.
        load : bool
            Whether to load the previously pre-processed dataset or pre-process from scratch.
            ``load`` should be False when we want to try different graph construction and
            featurization methods and need to preprocess from scratch. Default to True.
        log_every : bool
            Print a message every time ``log_every`` molecules are processed.
        """
        if os.path.exists(self.cache_file_path) and load:
            # DGLGraphs have been constructed before, reload them
            print('Loading previously saved dgl graphs...')
            self.graphs, label_dict = load_graphs(self.cache_file_path)
            self.labels = label_dict['labels']
            self.mask = label_dict['mask']
        else:
            print('Processing dgl graphs from scratch...')
            self.graphs = []
            for i, s in enumerate(self.smiles):
                if (i + 1) % log_every == 0:
                    print('Processing molecule {:d}/{:d}'.format(i+1, len(self)))
                self.graphs.append(smiles_to_graph(s, node_featurizer=node_featurizer,
                                                   edge_featurizer=edge_featurizer))
            _label_values = self.df[self.task_names].values
            # np.nan_to_num will also turn inf into a very large number
            self.labels = F.zerocopy_from_numpy(np.nan_to_num(_label_values).astype(np.float32))
            self.mask = F.zerocopy_from_numpy((~np.isnan(_label_values)).astype(np.float32))
            save_graphs(self.cache_file_path, self.graphs,
                        labels={'labels': self.labels, 'mask': self.mask})

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
        Tensor of dtype float32 and shape (T)
            Labels of the datapoint for all tasks
        Tensor of dtype float32 and shape (T)
            Binary masks indicating the existence of labels for all tasks
        """
        return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]

    def __len__(self):
        """Size for the dataset

        Returns
        -------
        int
            Size for the dataset
        """
        return len(self.smiles)
