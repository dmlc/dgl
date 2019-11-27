"""Utils for RDKit, adapted from DeepChem
(https://github.com/deepchem/deepchem/blob/master/deepchem)."""
import warnings

from multiprocessing import Pool

try:
    import pdbfixer
    import simtk

    from rdkit import Chem
    from rdkit.Chem import AllChem
    from StringIO import StringIO
except ImportError:
    from io import StringIO

__all__ = ['add_hydrogens_to_mol',
           'get_mol_3D_coordinates',
           'load_molecule',
           'multiprocess_load_molecule']

def add_hydrogens_to_mol(mol):
    """Add hydrogens to an RDKit molecule instance.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance.

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance with hydrogens added. For failures in adding hydrogens,
        the original RDKit molecule instance will be returned.
    """
    try:
        pdbblock = Chem.MolToPDBBlock(mol)
        pdb_stringio = StringIO()
        pdb_stringio.write(pdbblock)
        pdb_stringio.seek(0)

        fixer = pdbfixer.PDBFixer(pdbfile=pdb_stringio)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.4)

        hydrogenated_io = StringIO()
        simtk.openmm.app.PDBFile.writeFile(fixer.topology, fixer.positions,
                                           hydrogenated_io)
        hydrogenated_io.seek(0)
        mol = Chem.MolFromPDBBlock(hydrogenated_io.read(), sanitize=False, removeHs=False)
        pdb_stringio.close()
        hydrogenated_io.close()
    except ValueError:
        warnings.warn('Failed to add hydrogens to the molecule.')
    return mol

def get_mol_3D_coordinates(mol):
    """Get 3D coordinates of the molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance.

    Returns
    -------
    ndarray of shape (N, 3) or None
        The 3D coordinates of atoms in the molecule. N for the number of atoms in
        the molecule. For failures in getting the conformations, None will be returned.
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

def load_molecule(molecule_file, add_hydrogens=False,
                  sanitize=True, calc_charges=False, remove_hs=True):
    """Load a molecule from a file.

    Parameters
    ----------
    molecule_file : str
        Path to file for storing a molecule, which can be of format '.mol2', '.sdf',
        '.pdbqt', or '.pdb'.
    add_hydrogens : bool
        Whether to add hydrogens via pdbfixer. Default to True.
    sanitize : bool
        Whether sanitization is performed in initializing RDKit molecule instances. See
        https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
        Default to False.
    calc_charges : bool
        Whether to add Gasteiger charges via RDKit. Setting this to be True will enforce
        ``add_hydrogens`` and ``sanitize`` to be True. Default to True.
    remove_hs : bool
        Whether to remove hydrogens via RDKit. Default to True.

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the loaded molecule.
    coordinates : ndarray of shape (N, 3) or None
        The 3D coordinates of atoms in the molecule. N for the number of atoms in
        the molecule. For failures in getting the conformations, None will be returned.
    """
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as f:
            pdbqt_data = f.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        return ValueError('Expect the format of the molecule_file to be '
                          'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    if mol is None:
        return None, None

    if add_hydrogens or calc_charges:
        mol = add_hydrogens_to_mol(mol)

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

    coordinates = get_mol_3D_coordinates(mol)

    return mol, coordinates

def multiprocess_load_molecule(files, add_hydrogens=False, sanitize=True, calc_charges=False,
                               remove_hs=True, num_processes=None, msg=None, log_every_n=None):
    """"""
    if log_every_n is not None:
        assert msg is not None

    async_results = []
    with Pool(processes=num_processes) as pool:
        for f in files:
            async_results.append(pool.apply_async(
                load_molecule, (f, add_hydrogens, sanitize, calc_charges, remove_hs)))

    mols_loaded = []
    for i, result in enumerate(async_results):
        if (log_every_n is not None) and ((i + 1) % log_every_n == 0):
            print('Processing {} {:d}/{:d}'.format(msg, i+1, len(files)))
        mols_loaded.append(result.get())

    return mols_loaded
