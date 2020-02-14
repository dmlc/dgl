"""Utils for RDKit, mostly adapted from DeepChem
(https://github.com/deepchem/deepchem/blob/master/deepchem)."""
import warnings

from functools import partial
from multiprocessing import Pool

from ....contrib.deprecation import deprecated

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
           'multiprocess_load_molecules']

@deprecated('')
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

@deprecated('Import it from dgllife.utils.rdkit_utils instead.')
def get_mol_3D_coordinates(mol):
    """Get 3D coordinates of the molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance.

    Returns
    -------
    numpy.ndarray of shape (N, 3) or None
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

@deprecated('Import it from dgllife.utils.rdkit_utils instead.')
def load_molecule(molecule_file, add_hydrogens=False, sanitize=False, calc_charges=False,
                  remove_hs=False, use_conformation=True):
    """Load a molecule from a file.

    Parameters
    ----------
    molecule_file : str
        Path to file for storing a molecule, which can be of format '.mol2', '.sdf',
        '.pdbqt', or '.pdb'.
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

    try:
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
    except:
        return None, None

    if use_conformation:
        coordinates = get_mol_3D_coordinates(mol)
    else:
        coordinates = None

    return mol, coordinates

@deprecated('Import it from dgllife.utils.rdkit_utils instead.')
def multiprocess_load_molecules(files, add_hydrogens=False, sanitize=False, calc_charges=False,
                                remove_hs=False, use_conformation=True, num_processes=2):
    """Load molecules from files with multiprocessing.

    Parameters
    ----------
    files : list of str
        Each element is a path to a file storing a molecule, which can be of format '.mol2',
        '.sdf', '.pdbqt', or '.pdb'.
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
        for i, f in enumerate(files):
            mols_loaded.append(load_molecule(
                f, add_hydrogens=add_hydrogens, sanitize=sanitize, calc_charges=calc_charges,
                remove_hs=remove_hs, use_conformation=use_conformation))
    else:
        with Pool(processes=num_processes) as pool:
            mols_loaded = pool.map_async(partial(
                load_molecule, add_hydrogens=add_hydrogens, sanitize=sanitize,
                calc_charges=calc_charges, remove_hs=remove_hs,
                use_conformation=use_conformation), files)
            mols_loaded = mols_loaded.get()

    return mols_loaded
