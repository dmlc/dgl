"""Convert complexes into DGLHeteroGraphs"""
import numpy as np

from .... import graph, bipartite, hetero_from_relations
from .... import backend as F

try:
    import mdtraj
except ImportError:
    pass

__all__ = ['k_nearest_neighbors',
           'ACNN_graph_construction_and_featurization']

def filter_out_hydrogens(mol):
    """Return indices for non-hydrogen atoms.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance.
    """
    indices_left = []
    for i, atom in enumerate(mol.GetAtoms()):
        atomic_num = atom.GetAtomicNum()
        # Hydrogen atoms have an atomic number of 1.
        if atomic_num != 1:
            indices_left.append(i)
    return indices_left

def get_atomic_numbers(mol, indices):
    atomic_numbers = []
    for i in indices:
        atom = mol.GetAtomWithIdx(i)
        atomic_numbers.append(atom.GetAtomicNum())
    return atomic_numbers

def k_nearest_neighbors(coordinates, neighbor_cutoff, max_num_neighbors):
    """Find k nearest neighbors for each atom based on the 3D coordinates.

    Parameters
    ----------
    coordinates : numpy.ndarray of shape (N, 3)
        The 3D coordinates of atoms in the molecule. N for the number of atoms.
    neighbor_cutoff : float
        Distance cutoff to define 'neighboring'.
    max_num_neighbors : int or None.
        If not None, then this specifies the maximum number of closest neighbors
        allowed for each atom.

    Returns
    -------
    neighbor_list : dict(int -> list of ints)
        Mapping atom indices to their k nearest neighbors.
    """
    num_atoms = coordinates.shape[0]
    traj = mdtraj.Trajectory(coordinates.reshape((1, num_atoms, 3)), None)
    neighbors = mdtraj.geometry.compute_neighborlist(traj, neighbor_cutoff)
    srcs, dsts, distances = [], [], []
    for i in range(num_atoms):
        delta = coordinates[i] - coordinates.take(neighbors[i], axis=0)
        dist = np.linalg.norm(delta, axis=1)
        if max_num_neighbors is not None and len(neighbors[i]) > max_num_neighbors:
            sorted_neighbors = list(zip(dist, neighbors[i]))
            # Sort neighbors based on distance from smallest to largest
            sorted_neighbors.sort(key=lambda tup: tup[0])
            dsts.extend([i for _ in range(max_num_neighbors)])
            srcs.extend([int(sorted_neighbors[j][1]) for j in range(max_num_neighbors)])
            distances.extend([float(sorted_neighbors[j][0]) for j in range(max_num_neighbors)])
        else:
            dsts.extend([i for _ in range(len(neighbors[i]))])
            srcs.extend(neighbors[i].tolist())
            distances.extend(dist.tolist())

    return srcs, dsts, distances

def ACNN_graph_construction_and_featurization(protein_mol,
                                              ligand_mol,
                                              protein_coordinates,
                                              ligand_coordinates,
                                              neighbor_cutoff=12.,
                                              max_num_neighbors=12,
                                              strip_hydrogens=False):
    assert protein_coordinates is not None, 'Expect protein_coordinates to be provided.'
    assert ligand_coordinates is not None, 'Expect ligand_coordinates to be provided.'

    if strip_hydrogens:
        # Remove hydrogen atoms and their corresponding coordinates
        protein_atom_indices_left = filter_out_hydrogens(protein_mol)
        ligand_atom_indices_left = filter_out_hydrogens(ligand_mol)
        protein_coordinates = protein_coordinates.take(protein_atom_indices_left, axis=0)
        ligand_coordinates = ligand_coordinates.take(ligand_atom_indices_left, axis=0)
    else:
        protein_atom_indices_left = list(range(protein_mol.GetNumAtoms()))
        ligand_atom_indices_left = list(range(ligand_mol.GetNumAtoms()))

    # Construct graph for atoms in the protein
    protein_srcs, protein_dsts, protein_dists = k_nearest_neighbors(
        protein_coordinates, neighbor_cutoff, max_num_neighbors)
    protein_graph = graph((protein_srcs, protein_dsts),
                          'protein_atom', 'protein', len(protein_atom_indices_left))
    protein_graph.edata['distance'] = F.reshape(F.zerocopy_from_numpy(
        np.array(protein_dists).astype(np.float32)), (-1, 1))

    # Construct graph for atoms in the ligand
    ligand_srcs, ligand_dsts, ligand_dists = k_nearest_neighbors(
        ligand_coordinates, neighbor_cutoff, max_num_neighbors)
    ligand_graph = graph((ligand_srcs, ligand_dsts),
                         'ligand_atom', 'ligand', len(ligand_atom_indices_left))
    ligand_graph.edata['distance'] = F.reshape(F.zerocopy_from_numpy(
        np.array(ligand_dists).astype(np.float32)), (-1, 1))

    # Construct 4 graphs for complex representation, including the connection within
    # protein atoms, the connection within ligand atoms and the connection between
    # protein and ligand atoms.
    complex_srcs, complex_dsts, complex_dists = k_nearest_neighbors(
        np.concatenate([protein_coordinates, ligand_coordinates]),
        neighbor_cutoff, max_num_neighbors)
    complex_srcs = np.array(complex_srcs)
    complex_dsts = np.array(complex_dsts)
    complex_dists = np.array(complex_dists)
    offset = len(protein_atom_indices_left)

    # ('protein_atom', 'complex', 'protein_atom')
    inter_protein_indices = np.intersect1d(
        np.where(complex_srcs < offset)[0],
        np.where(complex_dsts < offset)[0],
        assume_unique=True)
    inter_protein_graph = graph(
        (complex_srcs[inter_protein_indices].tolist(),
         complex_dsts[inter_protein_indices].tolist()),
        'protein_atom', 'complex', len(protein_atom_indices_left))
    inter_protein_graph.edata['distance'] = F.reshape(F.zerocopy_from_numpy(
        complex_dists[inter_protein_indices].astype(np.float32)), (-1, 1))

    # ('ligand_atom', 'complex', 'ligand_atom')
    inter_ligand_indices = np.intersect1d(
        np.where(complex_srcs >= offset)[0],
        np.where(complex_dsts >= offset)[0],
        assume_unique=True)
    inter_ligand_graph = graph(
        ((complex_srcs[inter_ligand_indices] - offset).tolist(),
         (complex_dsts[inter_ligand_indices] - offset).tolist()),
        'ligand_atom', 'complex', len(ligand_atom_indices_left))
    inter_ligand_graph.edata['distance'] = F.reshape(F.zerocopy_from_numpy(
        complex_dists[inter_ligand_indices].astype(np.float32)), (-1, 1))

    # ('protein_atom', 'complex', 'ligand_atom')
    protein_ligand_indices = np.intersect1d(
        np.where(complex_srcs < offset)[0],
        np.where(complex_dsts >= offset)[0],
        assume_unique=True)
    protein_ligand_graph = bipartite(
        (complex_srcs[protein_ligand_indices].tolist(),
         (complex_dsts[protein_ligand_indices] - offset).tolist()),
        'protein_atom', 'complex', 'ligand_atom',
        (len(protein_atom_indices_left), len(ligand_atom_indices_left)))
    protein_ligand_graph.edata['distance'] = F.reshape(F.zerocopy_from_numpy(
        complex_dists[protein_ligand_indices].astype(np.float32)), (-1, 1))

    # ('ligand_atom', 'complex', 'protein_atom')
    ligand_protein_indices = np.intersect1d(
        np.where(complex_srcs >= offset)[0],
        np.where(complex_dsts < offset)[0],
        assume_unique=True)
    ligand_protein_graph = bipartite(
        ((complex_srcs[ligand_protein_indices] - offset).tolist(),
         complex_dsts[ligand_protein_indices].tolist()),
        'ligand_atom', 'complex', 'protein_atom',
        (len(ligand_atom_indices_left), len(protein_atom_indices_left)))
    ligand_protein_graph.edata['distance'] = F.reshape(F.zerocopy_from_numpy(
        complex_dists[ligand_protein_indices].astype(np.float32)), (-1, 1))

    # Get atomic numbers for all atoms left and set node features
    g = hetero_from_relations(
        [protein_graph,
         ligand_graph,
         inter_ligand_graph,
         inter_protein_graph,
         protein_ligand_graph,
         ligand_protein_graph]
    )
    protein_atomic_numbers = np.array(get_atomic_numbers(protein_mol, protein_atom_indices_left))
    ligand_atomic_numbers = np.array(get_atomic_numbers(ligand_mol, ligand_atom_indices_left))
    g.nodes['protein_atom'].data['atomic_number'] = F.reshape(F.zerocopy_from_numpy(
        protein_atomic_numbers.astype(np.float32)), (-1, 1))
    g.nodes['ligand_atom'].data['atomic_number'] =  F.reshape(F.zerocopy_from_numpy(
        ligand_atomic_numbers.astype(np.float32)), (-1, 1))

    return g
