"""Utils for RDKit, mostly adapted from DeepChem
(https://github.com/deepchem/deepchem/blob/master/deepchem)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import warnings

from functools import partial
from multiprocessing import Pool
from rdkit import Chem
from rdkit.Chem import AllChem

__all__ = ['get_mol_3d_coordinates',
           'load_molecule',
           'multiprocess_load_molecules']

# pylint: disable=W0702
def get_mol_3d_coordinates(mol):
    """Get 3D coordinates of the molecule.

    This function requires that molecular conformation has been initialized.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance.

    Returns
    -------
    numpy.ndarray of shape (N, 3) or None
        The 3D coordinates of atoms in the molecule. N for the number of atoms in
        the molecule. For failures in getting the conformations, None will be returned.

    Examples
    --------
    An error will occur in the example below since the molecule object does not
    carry conformation information.

    >>> from rdkit import Chem
    >>> from dgllife.utils import get_mol_3d_coordinates

    >>> mol = Chem.MolFromSmiles('CCO')

    Below we give a working example based on molecule conformation initialized from calculation.

    >>> from rdkit.Chem import AllChem
    >>> AllChem.EmbedMolecule(mol)
    >>> AllChem.MMFFOptimizeMolecule(mol)
    >>> coords = get_mol_3d_coordinates(mol)
    >>> print(coords)
    array([[ 1.20967478, -0.25802181,  0.        ],
           [-0.05021255,  0.57068079,  0.        ],
           [-1.15946223, -0.31265898,  0.        ]])
    """
    try:
        conf = mol.GetConformer()
        conf_num_atoms = conf.GetNumAtoms()
        mol_num_atoms = mol.GetNumAtoms()
        assert mol_num_atoms == conf_num_atoms, \
            'Expect the number of atoms in the molecule and its conformation ' \
            'to be the same, got {:d} and {:d}'.format(mol_num_atoms, conf_num_atoms)
        return conf.GetPositions()
    except:
        warnings.warn('Unable to get conformation of the molecule.')
        return None

# pylint: disable=E1101
def load_molecule(molecule_file, sanitize=False, calc_charges=False,
                  remove_hs=False, use_conformation=True):
    """Load a molecule from a file of format ``.mol2`` or ``.sdf`` or ``.pdbqt`` or ``.pdb``.

    Parameters
    ----------
    molecule_file : str
        Path to file for storing a molecule, which can be of format ``.mol2`` or ``.sdf``
        or ``.pdbqt`` or ``.pdb``.
    sanitize : bool
        Whether sanitization is performed in initializing RDKit molecule instances. See
        https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
        Default to False.
    calc_charges : bool
        Whether to add Gasteiger charges via RDKit. Setting this to be True will enforce
        ``sanitize`` to be True. Default to False.
    remove_hs : bool
        Whether to remove hydrogens via RDKit. Note that removing hydrogens can be quite
        slow for large molecules. Default to False.
    use_conformation : bool
        Whether we need to extract molecular conformation from proteins and ligands.
        Default to True.

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the loaded molecule.
    coordinates : np.ndarray of shape (N, 3) or None
        The 3D coordinates of atoms in the molecule. N for the number of atoms in
        the molecule. None will be returned if ``use_conformation`` is False or
        we failed to get conformation information.
    """
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        return ValueError('Expect the format of the molecule_file to be '
                          'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol)
    except:
        return None, None

    if use_conformation:
        coordinates = get_mol_3d_coordinates(mol)
    else:
        coordinates = None

    return mol, coordinates

def multiprocess_load_molecules(files, sanitize=False, calc_charges=False,
                                remove_hs=False, use_conformation=True, num_processes=2):
    """Load molecules from files with multiprocessing, which can be of format ``.mol2`` or
    ``.sdf`` or ``.pdbqt`` or ``.pdb``.

    Parameters
    ----------
    files : list of str
        Each element is a path to a file storing a molecule, which can be of format ``.mol2``,
        ``.sdf``, ``.pdbqt``, or ``.pdb``.
    sanitize : bool
        Whether sanitization is performed in initializing RDKit molecule instances. See
        https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
        Default to False.
    calc_charges : bool
        Whether to add Gasteiger charges via RDKit. Setting this to be True will enforce
        ``sanitize`` to be True. Default to False.
    remove_hs : bool
        Whether to remove hydrogens via RDKit. Note that removing hydrogens can be quite
        slow for large molecules. Default to False.
    use_conformation : bool
        Whether we need to extract molecular conformation from proteins and ligands.
        Default to True.
    num_processes : int or None
        Number of worker processes to use. If None,
        then we will use the number of CPUs in the systetm. Default to 2.

    Returns
    -------
    list of 2-tuples
        The first element of each 2-tuple is an RDKit molecule instance. The second element
        of each 2-tuple is the 3D atom coordinates of the corresponding molecule if
        use_conformation is True and the coordinates has been successfully loaded. Otherwise,
        it will be None.
    """
    if num_processes == 1:
        mols_loaded = []
        for f in files:
            mols_loaded.append(load_molecule(
                f, sanitize=sanitize, calc_charges=calc_charges,
                remove_hs=remove_hs, use_conformation=use_conformation))
    else:
        with Pool(processes=num_processes) as pool:
            mols_loaded = pool.map_async(partial(
                load_molecule, sanitize=sanitize, calc_charges=calc_charges,
                remove_hs=remove_hs, use_conformation=use_conformation), files)
            mols_loaded = mols_loaded.get()

    return mols_loaded
