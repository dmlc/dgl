
import os

from ..utils import get_download_dir, download, _get_dgl_url, extract_archive
from ... import backend as F

class PDBBind(object):
    """PDBbind dataset processed by MoleculeNet.
    The description below is mainly based on
    `[1] <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a#cit50>`__.
    The PDBbind database consists of experimentally measured binding affinities for
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
    subset_choice : str
        In MoleculeNet, we can use either the "refined" subset or the "core" subset. We can
        retrieve them by setting ``subset_choice`` to be ``'refined'`` or ``'core'``.
    """
    def __init__(self, subset_choice):
        self.task_names = ['-logKd/Ki']
        self.n_tasks = len(self.task_names)

        self._url = 'dataset/pdbbind_v2015.tar.gz'
        root_dir_path = get_download_dir()
        data_path = root_dir_path + '/pdbbind_v2015.tar.gz'
        extracted_data_path = root_dir_path + '/pdbbind_v2015'
        download(_get_dgl_url(self._url), path=data_path)
        extract_archive(data_path, extracted_data_path)

        if subset_choice == 'core':
            index_label_file = extracted_data_path + '/v2015/INDEX_core_data.2013'
        elif subset_choice == 'refined':
            index_label_file = extracted_data_path + '/v2015/INDEX_refined_data.2015'
        else:
            raise ValueError(
                'Expect the subset_choice to be either '
                'core or refined, got {}'.format(subset_choice))

        # Get the location of all ligand-protein pairs
        with open(index_label_file, 'r') as f:
            pdbs = [line[:4] for line in f.readlines() if line[0] != "#"]

        protein_files = [os.path.join(
            extracted_data_path, 'v2015', pdb, '{}_protein.pdb'.format(pdb)) for pdb in pdbs]
        ligand_files = [os.path.join(
            extracted_data_path, 'v2015', pdb, '{}_ligand.sdf'.format(pdb)) for pdb in pdbs]

        # Get labels
        with open(index_label_file, 'r') as f:
            labels = [float(line.split()[3]) for line in f.readlines() if line[0] != '#']
            self.labels = F.reshape(F.tensor(labels), (-1, 1))
