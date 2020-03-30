"""Convert molecules into DGLGraphs."""
import numpy as np

from functools import partial

from .... import DGLGraph
from ....contrib.deprecation import deprecated

try:
    import mdtraj
    from rdkit import Chem
    from rdkit.Chem import rdmolfiles, rdmolops
except ImportError:
    pass

__all__ = ['mol_to_graph',
           'smiles_to_bigraph',
           'mol_to_bigraph',
           'smiles_to_complete_graph',
           'mol_to_complete_graph',
           'k_nearest_neighbors']

@deprecated('Import it from dgllife.utils instead.')
def mol_to_graph(mol, graph_constructor, node_featurizer, edge_featurizer):
    """Convert an RDKit molecule object into a DGLGraph and featurize for it.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    graph_constructor : callable
        Takes an RDKit molecule as input and returns a DGLGraph
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to
        update ndata for a DGLGraph.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to
        update edata for a DGLGraph.

    Returns
    -------
    g : DGLGraph
        Converted DGLGraph for the molecule
    """
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    g = graph_constructor(mol)

    if node_featurizer is not None:
        g.ndata.update(node_featurizer(mol))

    if edge_featurizer is not None:
        g.edata.update(edge_featurizer(mol))

    return g

def construct_bigraph_from_mol(mol, add_self_loop=False):
    """Construct a bi-directed DGLGraph with topology only for the molecule.

    The **i** th atom in the molecule, i.e. ``mol.GetAtomWithIdx(i)``, corresponds to the
    **i** th node in the returned DGLGraph.

    The **i** th bond in the molecule, i.e. ``mol.GetBondWithIdx(i)``, corresponds to the
    **(2i)**-th and **(2i+1)**-th edges in the returned DGLGraph. The **(2i)**-th and
    **(2i+1)**-th edges will be separately from **u** to **v** and **v** to **u**, where
    **u** is ``bond.GetBeginAtomIdx()`` and **v** is ``bond.GetEndAtomIdx()``.

    If self loops are added, the last **n** edges will separately be self loops for
    atoms ``0, 1, ..., n-1``.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.

    Returns
    -------
    g : DGLGraph
        Empty bigraph topology of the molecule
    """
    g = DGLGraph()

    # Add nodes
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    # Add edges
    src_list = []
    dst_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])
    g.add_edges(src_list, dst_list)

    if add_self_loop:
        nodes = g.nodes()
        g.add_edges(nodes, nodes)

    return g

@deprecated('Import it from dgllife.utils instead.')
def mol_to_bigraph(mol, add_self_loop=False,
                   node_featurizer=None,
                   edge_featurizer=None):
    """Convert an RDKit molecule object into a bi-directed DGLGraph and featurize for it.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.

    Returns
    -------
    g : DGLGraph
        Bi-directed DGLGraph for the molecule
    """
    return mol_to_graph(mol, partial(construct_bigraph_from_mol, add_self_loop=add_self_loop),
                        node_featurizer, edge_featurizer)

@deprecated('Import it from dgllife.utils instead.')
def smiles_to_bigraph(smiles, add_self_loop=False,
                      node_featurizer=None,
                      edge_featurizer=None):
    """Convert a SMILES into a bi-directed DGLGraph and featurize for it.

    Parameters
    ----------
    smiles : str
        String of SMILES
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.

    Returns
    -------
    g : DGLGraph
        Bi-directed DGLGraph for the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol_to_bigraph(mol, add_self_loop, node_featurizer, edge_featurizer)

def construct_complete_graph_from_mol(mol, add_self_loop=False):
    """Construct a complete graph with topology only for the molecule

    The **i** th atom in the molecule, i.e. ``mol.GetAtomWithIdx(i)``, corresponds to the
    **i** th node in the returned DGLGraph.

    The edges are in the order of (0, 0), (1, 0), (2, 0), ... (0, 1), (1, 1), (2, 1), ...
    If self loops are not created, we will not have (0, 0), (1, 1), ...

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.

    Returns
    -------
    g : DGLGraph
        Empty complete graph topology of the molecule
    """
    g = DGLGraph()
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    if add_self_loop:
        g.add_edges(
            [i for i in range(num_atoms) for j in range(num_atoms)],
            [j for i in range(num_atoms) for j in range(num_atoms)])
    else:
        g.add_edges(
            [i for i in range(num_atoms) for j in range(num_atoms - 1)], [
                j for i in range(num_atoms)
                for j in range(num_atoms) if i != j
            ])

    return g

@deprecated('Import it from dgllife.utils instead.')
def mol_to_complete_graph(mol, add_self_loop=False,
                          node_featurizer=None,
                          edge_featurizer=None):
    """Convert an RDKit molecule into a complete DGLGraph and featurize for it.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.

    Returns
    -------
    g : DGLGraph
        Complete DGLGraph for the molecule
    """
    return mol_to_graph(mol, partial(construct_complete_graph_from_mol, add_self_loop=add_self_loop),
                        node_featurizer, edge_featurizer)

@deprecated('Import it from dgllife.utils instead.')
def smiles_to_complete_graph(smiles, add_self_loop=False,
                             node_featurizer=None,
                             edge_featurizer=None):
    """Convert a SMILES into a complete DGLGraph and featurize for it.

    Parameters
    ----------
    smiles : str
        String of SMILES
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.

    Returns
    -------
    g : DGLGraph
        Complete DGLGraph for the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol_to_complete_graph(mol, add_self_loop, node_featurizer, edge_featurizer)

@deprecated('Import it from dgllife.utils instead.')
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
    Returns
    -------
    srcs : list of int
        Source nodes.
    dsts : list of int
        Destination nodes.
    distances : list of float
        Distances between the end nodes.
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
