"""Convert molecules into DGLGraphs."""
# pylint: disable= no-member, arguments-differ, invalid-name
from functools import partial
import torch

from dgl import DGLGraph
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
from sklearn.neighbors import NearestNeighbors

__all__ = ['mol_to_graph',
           'smiles_to_bigraph',
           'mol_to_bigraph',
           'smiles_to_complete_graph',
           'mol_to_complete_graph',
           'k_nearest_neighbors',
           'mol_to_nearest_neighbor_graph',
           'smiles_to_nearest_neighbor_graph']

# pylint: disable=I1101
def mol_to_graph(mol, graph_constructor, node_featurizer, edge_featurizer, canonical_atom_order):
    """Convert an RDKit molecule object into a DGLGraph and featurize for it.

    This function can be used to construct any arbitrary ``DGLGraph`` from an
    RDKit molecule instance.

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
    canonical_atom_order : bool
        Whether to use a canonical order of atoms returned by RDKit. Setting it
        to true might change the order of atoms in the graph constructed.

    Returns
    -------
    g : DGLGraph
        Converted DGLGraph for the molecule

    See Also
    --------
    mol_to_bigraph
    mol_to_complete_graph
    mol_to_nearest_neighbor_graph
    """
    if canonical_atom_order:
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

def mol_to_bigraph(mol, add_self_loop=False,
                   node_featurizer=None,
                   edge_featurizer=None,
                   canonical_atom_order=True):
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
    canonical_atom_order : bool
        Whether to use a canonical order of atoms returned by RDKit. Setting it
        to true might change the order of atoms in the graph constructed. Default
        to True.

    Returns
    -------
    g : DGLGraph
        Bi-directed DGLGraph for the molecule

    Examples
    --------
    >>> from rdkit import Chem
    >>> from dgllife.utils import mol_to_bigraph

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> g = mol_to_bigraph(mol)
    >>> print(g)
    DGLGraph(num_nodes=3, num_edges=4,
             ndata_schemes={}
             edata_schemes={})

    We can also initialize node/edge features when constructing graphs.

    >>> import torch
    >>> from rdkit import Chem
    >>> from dgllife.utils import mol_to_bigraph

    >>> def featurize_atoms(mol):
    >>>     feats = []
    >>>     for atom in mol.GetAtoms():
    >>>         feats.append(atom.GetAtomicNum())
    >>>     return {'atomic': torch.tensor(feats).reshape(-1, 1).float()}

    >>> def featurize_bonds(mol):
    >>>     feats = []
    >>>     bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
    >>>                   Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    >>>     for bond in mol.GetBonds():
    >>>         btype = bond_types.index(bond.GetBondType())
    >>>         # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
    >>>         feats.extend([btype, btype])
    >>>     return {'type': torch.tensor(feats).reshape(-1, 1).float()}

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> g = mol_to_bigraph(mol, node_featurizer=featurize_atoms,
    >>>                    edge_featurizer=featurize_bonds)
    >>> print(g.ndata['atomic'])
    tensor([[6.],
            [8.],
            [6.]])
    >>> print(g.edata['type'])
    tensor([[0.],
            [0.],
            [0.],
            [0.]])

    See Also
    --------
    smiles_to_bigraph
    """
    return mol_to_graph(mol, partial(construct_bigraph_from_mol, add_self_loop=add_self_loop),
                        node_featurizer, edge_featurizer, canonical_atom_order)

def smiles_to_bigraph(smiles, add_self_loop=False,
                      node_featurizer=None,
                      edge_featurizer=None,
                      canonical_atom_order=True):
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
    canonical_atom_order : bool
        Whether to use a canonical order of atoms returned by RDKit. Setting it
        to true might change the order of atoms in the graph constructed. Default
        to True.

    Returns
    -------
    g : DGLGraph
        Bi-directed DGLGraph for the molecule

    Examples
    --------
    >>> from dgllife.utils import smiles_to_bigraph

    >>> g = smiles_to_bigraph('CCO')
    >>> print(g)
    DGLGraph(num_nodes=3, num_edges=4,
             ndata_schemes={}
             edata_schemes={})

    We can also initialize node/edge features when constructing graphs.

    >>> import torch
    >>> from rdkit import Chem
    >>> from dgllife.utils import smiles_to_bigraph

    >>> def featurize_atoms(mol):
    >>>     feats = []
    >>>     for atom in mol.GetAtoms():
    >>>         feats.append(atom.GetAtomicNum())
    >>>     return {'atomic': torch.tensor(feats).reshape(-1, 1).float()}

    >>> def featurize_bonds(mol):
    >>>     feats = []
    >>>     bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
    >>>                   Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    >>>     for bond in mol.GetBonds():
    >>>         btype = bond_types.index(bond.GetBondType())
    >>>         # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
    >>>         feats.extend([btype, btype])
    >>>     return {'type': torch.tensor(feats).reshape(-1, 1).float()}

    >>> g = smiles_to_bigraph('CCO', node_featurizer=featurize_atoms,
    >>>                       edge_featurizer=featurize_bonds)
    >>> print(g.ndata['atomic'])
    tensor([[6.],
            [8.],
            [6.]])
    >>> print(g.edata['type'])
    tensor([[0.],
            [0.],
            [0.],
            [0.]])

    See Also
    --------
    mol_to_bigraph
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol_to_bigraph(mol, add_self_loop, node_featurizer,
                          edge_featurizer, canonical_atom_order)

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
    num_atoms = mol.GetNumAtoms()
    edge_list = []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j or add_self_loop:
                edge_list.append((i, j))
    g = DGLGraph(edge_list)

    return g

def mol_to_complete_graph(mol, add_self_loop=False,
                          node_featurizer=None,
                          edge_featurizer=None,
                          canonical_atom_order=True):
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
    canonical_atom_order : bool
        Whether to use a canonical order of atoms returned by RDKit. Setting it
        to true might change the order of atoms in the graph constructed. Default
        to True.

    Returns
    -------
    g : DGLGraph
        Complete DGLGraph for the molecule

    Examples
    --------
    >>> from rdkit import Chem
    >>> from dgllife.utils import mol_to_complete_graph

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> g = mol_to_complete_graph(mol)
    >>> print(g)
    DGLGraph(num_nodes=3, num_edges=6,
             ndata_schemes={}
             edata_schemes={})

    We can also initialize node/edge features when constructing graphs.

    >>> import torch
    >>> from rdkit import Chem
    >>> from dgllife.utils import mol_to_complete_graph
    >>> from functools import partial

    >>> def featurize_atoms(mol):
    >>>     feats = []
    >>>     for atom in mol.GetAtoms():
    >>>         feats.append(atom.GetAtomicNum())
    >>>     return {'atomic': torch.tensor(feats).reshape(-1, 1).float()}

    >>> def featurize_edges(mol, add_self_loop=False):
    >>>     feats = []
    >>>     num_atoms = mol.GetNumAtoms()
    >>>     atoms = list(mol.GetAtoms())
    >>>     distance_matrix = Chem.GetDistanceMatrix(mol)
    >>>     for i in range(num_atoms):
    >>>         for j in range(num_atoms):
    >>>             if i != j or add_self_loop:
    >>>                 feats.append(float(distance_matrix[i, j]))
    >>>     return {'dist': torch.tensor(feats).reshape(-1, 1).float()}

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> add_self_loop = True
    >>> g = mol_to_complete_graph(
    >>>         mol, add_self_loop=add_self_loop, node_featurizer=featurize_atoms,
    >>>         edge_featurizer=partial(featurize_edges, add_self_loop=add_self_loop))
    >>> print(g.ndata['atomic'])
    tensor([[6.],
            [8.],
            [6.]])
    >>> print(g.edata['dist'])
    tensor([[0.],
            [2.],
            [1.],
            [2.],
            [0.],
            [1.],
            [1.],
            [1.],
            [0.]])

    See Also
    --------
    smiles_to_complete_graph
    """
    return mol_to_graph(mol,
                        partial(construct_complete_graph_from_mol, add_self_loop=add_self_loop),
                        node_featurizer, edge_featurizer, canonical_atom_order)

def smiles_to_complete_graph(smiles, add_self_loop=False,
                             node_featurizer=None,
                             edge_featurizer=None,
                             canonical_atom_order=True):
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
    canonical_atom_order : bool
        Whether to use a canonical order of atoms returned by RDKit. Setting it
        to true might change the order of atoms in the graph constructed. Default
        to True.

    Returns
    -------
    g : DGLGraph
        Complete DGLGraph for the molecule

    Examples
    --------
    >>> from dgllife.utils import smiles_to_complete_graph

    >>> g = smiles_to_complete_graph('CCO')
    >>> print(g)
    DGLGraph(num_nodes=3, num_edges=6,
             ndata_schemes={}
             edata_schemes={})

    We can also initialize node/edge features when constructing graphs.

    >>> import torch
    >>> from rdkit import Chem
    >>> from dgllife.utils import smiles_to_complete_graph
    >>> from functools import partial

    >>> def featurize_atoms(mol):
    >>>     feats = []
    >>>     for atom in mol.GetAtoms():
    >>>         feats.append(atom.GetAtomicNum())
    >>>     return {'atomic': torch.tensor(feats).reshape(-1, 1).float()}

    >>> def featurize_edges(mol, add_self_loop=False):
    >>>     feats = []
    >>>     num_atoms = mol.GetNumAtoms()
    >>>     atoms = list(mol.GetAtoms())
    >>>     distance_matrix = Chem.GetDistanceMatrix(mol)
    >>>     for i in range(num_atoms):
    >>>         for j in range(num_atoms):
    >>>             if i != j or add_self_loop:
    >>>                 feats.append(float(distance_matrix[i, j]))
    >>>     return {'dist': torch.tensor(feats).reshape(-1, 1).float()}

    >>> add_self_loop = True
    >>> g = smiles_to_complete_graph(
    >>>         'CCO', add_self_loop=add_self_loop, node_featurizer=featurize_atoms,
    >>>         edge_featurizer=partial(featurize_edges, add_self_loop=add_self_loop))
    >>> print(g.ndata['atomic'])
    tensor([[6.],
            [8.],
            [6.]])
    >>> print(g.edata['dist'])
    tensor([[0.],
            [2.],
            [1.],
            [2.],
            [0.],
            [1.],
            [1.],
            [1.],
            [0.]])

    See Also
    --------
    mol_to_complete_graph
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol_to_complete_graph(mol, add_self_loop, node_featurizer,
                                 edge_featurizer, canonical_atom_order)

def k_nearest_neighbors(coordinates, neighbor_cutoff, max_num_neighbors=None,
                        p_distance=2, self_loops=False):
    """Find k nearest neighbors for each atom

    We do not guarantee that the edges are sorted according to the distance
    between atoms.

    Parameters
    ----------
    coordinates : numpy.ndarray of shape (N, D)
        The coordinates of atoms in the molecule. N for the number of atoms
        and D for the dimensions of the coordinates.
    neighbor_cutoff : float
        If the distance between a pair of nodes is larger than neighbor_cutoff,
        they will not be considered as neighboring nodes.
    max_num_neighbors : int or None.
        If not None, then this specifies the maximum number of neighbors
        allowed for each atom. Default to None.
    p_distance : int
        We compute the distance between neighbors using Minkowski (:math:`l_p`)
        distance. When ``p_distance = 1``, Minkowski distance is equivalent to
        Manhattan distance. When ``p_distance = 2``, Minkowski distance is
        equivalent to the standard Euclidean distance. Default to 2.
    self_loops : bool
        Whether to allow a node to be its own neighbor. Default to False.

    Returns
    -------
    srcs : list of int
        Source nodes.
    dsts : list of int
        Destination nodes, corresponding to ``srcs``.
    distances : list of float
        Distances between the end nodes, corresponding to ``srcs`` and ``dsts``.

    Examples
    --------
    >>> from dgllife.utils import get_mol_3d_coordinates, k_nearest_neighbors
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem

    >>> mol = Chem.MolFromSmiles('CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C')
    >>> AllChem.EmbedMolecule(mol)
    >>> AllChem.MMFFOptimizeMolecule(mol)
    >>> coords = get_mol_3d_coordinates(mol)
    >>> srcs, dsts, dists = k_nearest_neighbors(coords, neighbor_cutoff=1.25)
    >>> print(srcs)
    [8, 7, 11, 10, 20, 19]
    >>> print(dsts)
    [7, 8, 10, 11, 19, 20]
    >>> print(dists)
    [1.2084666104583117, 1.2084666104583117, 1.226457824344217,
     1.226457824344217, 1.2230522248065987, 1.2230522248065987]

    See Also
    --------
    get_mol_3d_coordinates
    mol_to_nearest_neighbor_graph
    smiles_to_nearest_neighbor_graph
    """
    num_atoms = coordinates.shape[0]
    model = NearestNeighbors(radius=neighbor_cutoff, p=p_distance)
    model.fit(coordinates)
    dists_, nbrs = model.radius_neighbors(coordinates)
    srcs, dsts, dists = [], [], []
    for i in range(num_atoms):
        dists_i = dists_[i].tolist()
        nbrs_i = nbrs[i].tolist()
        if not self_loops:
            dists_i.remove(0)
            nbrs_i.remove(i)
        if max_num_neighbors is not None and len(nbrs_i) > max_num_neighbors:
            packed_nbrs = list(zip(dists_i, nbrs_i))
            # Sort neighbors based on distance from smallest to largest
            packed_nbrs.sort(key=lambda tup: tup[0])
            dists_i, nbrs_i = map(list, zip(*packed_nbrs))
            dsts.extend([i for _ in range(max_num_neighbors)])
            srcs.extend(nbrs_i[:max_num_neighbors])
            dists.extend(dists_i[:max_num_neighbors])
        else:
            dsts.extend([i for _ in range(len(nbrs_i))])
            srcs.extend(nbrs_i)
            dists.extend(dists_i)

    return srcs, dsts, dists

# pylint: disable=E1102
def mol_to_nearest_neighbor_graph(mol,
                                  coordinates,
                                  neighbor_cutoff,
                                  max_num_neighbors=None,
                                  p_distance=2,
                                  add_self_loop=False,
                                  node_featurizer=None,
                                  edge_featurizer=None,
                                  canonical_atom_order=True,
                                  keep_dists=False,
                                  dist_field='dist'):
    """Convert an RDKit molecule into a nearest neighbor graph and featurize for it.

    Different from bigraph and complete graph, the nearest neighbor graph
    may not be symmetric since i is the closest neighbor of j does not
    necessarily suggest the other way.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    coordinates : numpy.ndarray of shape (N, D)
        The coordinates of atoms in the molecule. N for the number of atoms
        and D for the dimensions of the coordinates.
    neighbor_cutoff : float
        If the distance between a pair of nodes is larger than neighbor_cutoff,
        they will not be considered as neighboring nodes.
    max_num_neighbors : int or None.
        If not None, then this specifies the maximum number of neighbors
        allowed for each atom. Default to None.
    p_distance : int
        We compute the distance between neighbors using Minkowski (:math:`l_p`)
        distance. When ``p_distance = 1``, Minkowski distance is equivalent to
        Manhattan distance. When ``p_distance = 2``, Minkowski distance is
        equivalent to the standard Euclidean distance. Default to 2.
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.
    canonical_atom_order : bool
        Whether to use a canonical order of atoms returned by RDKit. Setting it
        to true might change the order of atoms in the graph constructed. Default
        to True.
    keep_dists : bool
        Whether to store the distance between neighboring atoms in ``edata`` of the
        constructed DGLGraphs. Default to False.
    dist_field : str
        Field for storing distance between neighboring atoms in ``edata``. This comes
        into effect only when ``keep_dists=True``. Default to ``'dist'``.

    Returns
    -------
    g : DGLGraph
        Nearest neighbor DGLGraph for the molecule

    Examples
    --------
    >>> from dgllife.utils import mol_to_nearest_neighbor_graph
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem

    >>> mol = Chem.MolFromSmiles('CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C')
    >>> AllChem.EmbedMolecule(mol)
    >>> AllChem.MMFFOptimizeMolecule(mol)
    >>> coords = get_mol_3d_coordinates(mol)
    >>> g = mol_to_nearest_neighbor_graph(mol, coords, neighbor_cutoff=1.25)
    >>> print(g)
    DGLGraph(num_nodes=23, num_edges=6,
             ndata_schemes={}
             edata_schemes={})

    Quite often we will want to use the distance between end atoms of edges, this can be
    achieved with

    >>> g = mol_to_nearest_neighbor_graph(mol, coords, neighbor_cutoff=1.25, keep_dists=True)
    >>> print(g.edata['dist'])
    tensor([[1.2024],
            [1.2024],
            [1.2270],
            [1.2270],
            [1.2259],
            [1.2259]])

    See Also
    --------
    get_mol_3d_coordinates
    k_nearest_neighbors
    smiles_to_nearest_neighbor_graph
    """
    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)

    srcs, dsts, dists = k_nearest_neighbors(coordinates=coordinates,
                                            neighbor_cutoff=neighbor_cutoff,
                                            max_num_neighbors=max_num_neighbors,
                                            p_distance=p_distance,
                                            self_loops=add_self_loop)
    g = DGLGraph()

    # Add nodes first since some nodes may be completely isolated
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    # Add edges
    g.add_edges(srcs, dsts)

    if node_featurizer is not None:
        g.ndata.update(node_featurizer(mol))

    if edge_featurizer is not None:
        g.edata.update(edge_featurizer(mol))

    if keep_dists:
        assert dist_field not in g.edata, \
            'Expect {} to be reserved for distance between neighboring atoms.'
        g.edata[dist_field] = torch.tensor(dists).float().reshape(-1, 1)

    return g

def smiles_to_nearest_neighbor_graph(smiles,
                                     coordinates,
                                     neighbor_cutoff,
                                     max_num_neighbors=None,
                                     p_distance=2,
                                     add_self_loop=False,
                                     node_featurizer=None,
                                     edge_featurizer=None,
                                     canonical_atom_order=True,
                                     keep_dists=False,
                                     dist_field='dist'):
    """Convert a SMILES into a nearest neighbor graph and featurize for it.

    Different from bigraph and complete graph, the nearest neighbor graph
    may not be symmetric since i is the closest neighbor of j does not
    necessarily suggest the other way.

    Parameters
    ----------
    smiles : str
        String of SMILES
    coordinates : numpy.ndarray of shape (N, D)
        The coordinates of atoms in the molecule. N for the number of atoms
        and D for the dimensions of the coordinates.
    neighbor_cutoff : float
        If the distance between a pair of nodes is larger than neighbor_cutoff,
        they will not be considered as neighboring nodes.
    max_num_neighbors : int or None.
        If not None, then this specifies the maximum number of neighbors
        allowed for each atom. Default to None.
    p_distance : int
        We compute the distance between neighbors using Minkowski (:math:`l_p`)
        distance. When ``p_distance = 1``, Minkowski distance is equivalent to
        Manhattan distance. When ``p_distance = 2``, Minkowski distance is
        equivalent to the standard Euclidean distance. Default to 2.
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.
    canonical_atom_order : bool
        Whether to use a canonical order of atoms returned by RDKit. Setting it
        to true might change the order of atoms in the graph constructed. Default
        to True.
    keep_dists : bool
        Whether to store the distance between neighboring atoms in ``edata`` of the
        constructed DGLGraphs. Default to False.
    dist_field : str
        Field for storing distance between neighboring atoms in ``edata``. This comes
        into effect only when ``keep_dists=True``. Default to ``'dist'``.

    Returns
    -------
    g : DGLGraph
        Nearest neighbor DGLGraph for the molecule

    Examples
    --------
    >>> from dgllife.utils import smiles_to_nearest_neighbor_graph
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem

    >>> smiles = 'CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C'
    >>> mol = Chem.MolFromSmiles(smiles)
    >>> AllChem.EmbedMolecule(mol)
    >>> AllChem.MMFFOptimizeMolecule(mol)
    >>> coords = get_mol_3d_coordinates(mol)
    >>> g = mol_to_nearest_neighbor_graph(mol, coords, neighbor_cutoff=1.25)
    >>> print(g)
    DGLGraph(num_nodes=23, num_edges=6,
             ndata_schemes={}
             edata_schemes={})

    Quite often we will want to use the distance between end atoms of edges, this can be
    achieved with

    >>> g = smiles_to_nearest_neighbor_graph(smiles, coords, neighbor_cutoff=1.25, keep_dists=True)
    >>> print(g.edata['dist'])
    tensor([[1.2024],
            [1.2024],
            [1.2270],
            [1.2270],
            [1.2259],
            [1.2259]])

    See Also
    --------
    get_mol_3d_coordinates
    k_nearest_neighbors
    mol_to_nearest_neighbor_graph
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol_to_nearest_neighbor_graph(
        mol, coordinates, neighbor_cutoff, max_num_neighbors, p_distance, add_self_loop,
        node_featurizer, edge_featurizer, canonical_atom_order, keep_dists, dist_field)
