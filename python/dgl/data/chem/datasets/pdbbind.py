"""PDBBind dataset processed by MoleculeNet."""
import numpy as np
import os
import pandas as pd

from ..utils import multiprocess_load_molecules, ACNN_graph_construction_and_featurization
from ...utils import get_download_dir, download, _get_dgl_url, extract_archive
from .... import backend as F
from ....contrib.deprecation import deprecated

class PDBBind(object):
    """PDBbind dataset processed by MoleculeNet.

    The description below is mainly based on
    `[1] <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a#cit50>`__.
    The PDBBind database consists of experimentally measured binding affinities for
    bio-molecular complexes `[2] <https://www.ncbi.nlm.nih.gov/pubmed/?term=15163179%5Buid%5D>`__,
    `[3] <https://www.ncbi.nlm.nih.gov/pubmed/?term=15943484%5Buid%5D>`__. It provides detailed
    3D Cartesian coordinates of both ligands and their target proteins derived from experimental
    (e.g., X-ray crystallography) measurements. The availability of coordinates of the
    protein-ligand complexes permits structure-based featurization that is aware of the
    protein-ligand binding geometry. The authors of
    `[1] <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a#cit50>`__ use the
    "refined" and "core" subsets of the database
    `[4] <https://www.ncbi.nlm.nih.gov/pubmed/?term=25301850%5Buid%5D>`__, more carefully
    processed for data artifacts, as additional benchmarking targets.

    References:

        * [1] MoleculeNet: a benchmark for molecular machine learning
        * [2] The PDBbind database: collection of binding affinities for protein-ligand complexes
        with known three-dimensional structures
        * [3] The PDBbind database: methodologies and updates
        * [4] PDB-wide collection of binding data: current status of the PDBbind database

    Parameters
    ----------
    subset : str
        In MoleculeNet, we can use either the "refined" subset or the "core" subset. We can
        retrieve them by setting ``subset`` to be ``'refined'`` or ``'core'``. The size
        of the ``'core'`` set is 195 and the size of the ``'refined'`` set is 3706.
    load_binding_pocket : bool
        Whether to load binding pockets or full proteins. Default to True.
    add_hydrogens : bool
        Whether to add hydrogens via pdbfixer. Default to False.
    sanitize : bool
        Whether sanitization is performed in initializing RDKit molecule instances. See
        https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
        Default to False.
    calc_charges : bool
        Whether to add Gasteiger charges via RDKit. Setting this to be True will enforce
        ``add_hydrogens`` and ``sanitize`` to be True. Default to False.
    remove_hs : bool
        Whether to remove hydrogens via RDKit. Note that removing hydrogens can be quite
        slow for large molecules. Default to False.
    use_conformation : bool
        Whether we need to extract molecular conformation from proteins and ligands.
        Default to True.
    construct_graph_and_featurize : callable
        Construct a DGLHeteroGraph for the use of GNNs. Mapping self.ligand_mols[i],
        self.protein_mols[i], self.ligand_coordinates[i] and self.protein_coordinates[i]
        to a DGLHeteroGraph. Default to :func:`ACNN_graph_construction_and_featurization`.
    zero_padding : bool
        Whether to perform zero padding. While DGL does not necessarily require zero padding,
        pooling operations for variable length inputs can introduce stochastic behaviour, which
        is not desired for sensitive scenarios. Default to True.
    num_processes : int or None
        Number of worker processes to use. If None,
        then we will use the number of CPUs in the system. Default to 64.
    """
    @deprecated('Import PDBBind from dgllife.data instead.', 'class')
    def __init__(self, subset, load_binding_pocket=True, add_hydrogens=False,
                 sanitize=False, calc_charges=False, remove_hs=False, use_conformation=True,
                 construct_graph_and_featurize=ACNN_graph_construction_and_featurization,
                 zero_padding=True, num_processes=64):
        self.task_names = ['-logKd/Ki']
        self.n_tasks = len(self.task_names)

        self._url = 'dataset/pdbbind_v2015.tar.gz'
        root_dir_path = get_download_dir()
        data_path = root_dir_path + '/pdbbind_v2015.tar.gz'
        extracted_data_path = root_dir_path + '/pdbbind_v2015'
        download(_get_dgl_url(self._url), path=data_path)
        extract_archive(data_path, extracted_data_path)

        if subset == 'core':
            index_label_file = extracted_data_path + '/v2015/INDEX_core_data.2013'
        elif subset == 'refined':
            index_label_file = extracted_data_path + '/v2015/INDEX_refined_data.2015'
        else:
            raise ValueError(
                'Expect the subset_choice to be either '
                'core or refined, got {}'.format(subset))

        self._preprocess(extracted_data_path, index_label_file, load_binding_pocket,
                         add_hydrogens, sanitize, calc_charges, remove_hs, use_conformation,
                         construct_graph_and_featurize, zero_padding, num_processes)

    def _filter_out_invalid(self, ligands_loaded, proteins_loaded, use_conformation):
        """Filter out invalid ligand-protein pairs.

        Parameters
        ----------
        ligands_loaded : list
            Each element is a 2-tuple of the RDKit molecule instance and its associated atom
            coordinates. None is used to represent invalid/non-existing molecule or coordinates.
        proteins_loaded : list
            Each element is a 2-tuple of the RDKit molecule instance and its associated atom
            coordinates. None is used to represent invalid/non-existing molecule or coordinates.
        use_conformation : bool
            Whether we need conformation information (atom coordinates) and filter out molecules
            without valid conformation.
        """
        num_pairs = len(proteins_loaded)
        self.indices, self.ligand_mols, self.protein_mols = [], [], []
        if use_conformation:
            self.ligand_coordinates, self.protein_coordinates = [], []
        else:
            # Use None for placeholders.
            self.ligand_coordinates = [None for _ in range(num_pairs)]
            self.protein_coordinates = [None for _ in range(num_pairs)]

        for i in range(num_pairs):
            ligand_mol, ligand_coordinates = ligands_loaded[i]
            protein_mol, protein_coordinates = proteins_loaded[i]
            if (not use_conformation) and all(v is not None for v in [protein_mol, ligand_mol]):
                self.indices.append(i)
                self.ligand_mols.append(ligand_mol)
                self.protein_mols.append(protein_mol)
            elif all(v is not None for v in [
                protein_mol, protein_coordinates, ligand_mol, ligand_coordinates]):
                self.indices.append(i)
                self.ligand_mols.append(ligand_mol)
                self.ligand_coordinates.append(ligand_coordinates)
                self.protein_mols.append(protein_mol)
                self.protein_coordinates.append(protein_coordinates)

    def _preprocess(self, root_path, index_label_file, load_binding_pocket,
                    add_hydrogens, sanitize, calc_charges, remove_hs, use_conformation,
                    construct_graph_and_featurize, zero_padding, num_processes):
        """Preprocess the dataset.

        The pre-processing proceeds as follows:

        1. Load the dataset
        2. Clean the dataset and filter out invalid pairs
        3. Construct graphs
        4. Prepare node and edge features

        Parameters
        ----------
        root_path : str
            Root path for molecule files.
        index_label_file : str
            Path to the index file for the dataset.
        load_binding_pocket : bool
            Whether to load binding pockets or full proteins.
        add_hydrogens : bool
            Whether to add hydrogens via pdbfixer.
        sanitize : bool
            Whether sanitization is performed in initializing RDKit molecule instances. See
            https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
        calc_charges : bool
            Whether to add Gasteiger charges via RDKit. Setting this to be True will enforce
            ``add_hydrogens`` and ``sanitize`` to be True.
        remove_hs : bool
            Whether to remove hydrogens via RDKit. Note that removing hydrogens can be quite
            slow for large molecules.
        use_conformation : bool
            Whether we need to extract molecular conformation from proteins and ligands.
        construct_graph_and_featurize : callable
            Construct a DGLHeteroGraph for the use of GNNs. Mapping self.ligand_mols[i],
            self.protein_mols[i], self.ligand_coordinates[i] and self.protein_coordinates[i]
            to a DGLHeteroGraph. Default to :func:`ACNN_graph_construction_and_featurization`.
        zero_padding : bool
            Whether to perform zero padding. While DGL does not necessarily require zero padding,
            pooling operations for variable length inputs can introduce stochastic behaviour, which
            is not desired for sensitive scenarios.
        num_processes : int or None
            Number of worker processes to use. If None,
            then we will use the number of CPUs in the system.
        """
        contents = []
        with open(index_label_file, 'r') as f:
            for line in f.readlines():
                if line[0] != "#":
                    splitted_elements = line.split()
                    if len(splitted_elements) == 8:
                        # Ignore "//"
                        contents.append(splitted_elements[:5] + splitted_elements[6:])
                    else:
                        print('Incorrect data format.')
                        print(splitted_elements)
        self.df = pd.DataFrame(contents, columns=(
            'PDB_code', 'resolution', 'release_year',
            '-logKd/Ki', 'Kd/Ki', 'reference', 'ligand_name'))
        pdbs = self.df['PDB_code'].tolist()

        self.ligand_files = [os.path.join(
            root_path, 'v2015', pdb, '{}_ligand.sdf'.format(pdb)) for pdb in pdbs]
        if load_binding_pocket:
            self.protein_files = [os.path.join(
                root_path, 'v2015', pdb, '{}_pocket.pdb'.format(pdb)) for pdb in pdbs]
        else:
            self.protein_files = [os.path.join(
                root_path, 'v2015', pdb, '{}_protein.pdb'.format(pdb)) for pdb in pdbs]

        num_processes = min(num_processes, len(pdbs))

        print('Loading ligands...')
        ligands_loaded = multiprocess_load_molecules(self.ligand_files,
                                                     add_hydrogens=add_hydrogens,
                                                     sanitize=sanitize,
                                                     calc_charges=calc_charges,
                                                     remove_hs=remove_hs,
                                                     use_conformation=use_conformation,
                                                     num_processes=num_processes)

        print('Loading proteins...')
        proteins_loaded = multiprocess_load_molecules(self.protein_files,
                                                      add_hydrogens=add_hydrogens,
                                                      sanitize=sanitize,
                                                      calc_charges=calc_charges,
                                                      remove_hs=remove_hs,
                                                      use_conformation=use_conformation,
                                                      num_processes=num_processes)

        self._filter_out_invalid(ligands_loaded, proteins_loaded, use_conformation)
        self.df = self.df.iloc[self.indices]
        self.labels = F.zerocopy_from_numpy(self.df[self.task_names].values.astype(np.float32))
        print('Finished cleaning the dataset, '
              'got {:d}/{:d} valid pairs'.format(len(self), len(pdbs)))

        # Prepare zero padding
        if zero_padding:
            max_num_ligand_atoms = 0
            max_num_protein_atoms = 0
            for i in range(len(self)):
                max_num_ligand_atoms = max(
                    max_num_ligand_atoms, self.ligand_mols[i].GetNumAtoms())
                max_num_protein_atoms = max(
                    max_num_protein_atoms, self.protein_mols[i].GetNumAtoms())
        else:
            max_num_ligand_atoms = None
            max_num_protein_atoms = None

        print('Start constructing graphs and featurizing them.')
        self.graphs = []
        for i in range(len(self)):
            print('Constructing and featurizing datapoint {:d}/{:d}'.format(i+1, len(self)))
            self.graphs.append(construct_graph_and_featurize(
                self.ligand_mols[i], self.protein_mols[i],
                self.ligand_coordinates[i], self.protein_coordinates[i],
                max_num_ligand_atoms, max_num_protein_atoms))

    def __len__(self):
        """Get the size of the dataset.

        Returns
        -------
        int
            Number of valid ligand-protein pairs in the dataset.
        """
        return len(self.indices)

    def __getitem__(self, item):
        """Get the datapoint associated with the index.

        Parameters
        ----------
        item : int
            Index for the datapoint.

        Returns
        -------
        int
            Index for the datapoint.
        rdkit.Chem.rdchem.Mol
            RDKit molecule instance for the ligand molecule.
        rdkit.Chem.rdchem.Mol
            RDKit molecule instance for the protein molecule.
        DGLHeteroGraph
            Pre-processed DGLHeteroGraph with features extracted.
        Float32 tensor
            Label for the datapoint.
        """
        return item, self.ligand_mols[item], self.protein_mols[item], \
               self.graphs[item], self.labels[item]
