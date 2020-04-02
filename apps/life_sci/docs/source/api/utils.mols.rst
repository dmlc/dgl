.. _apiutilsmols:

Utils for Molecules
===================

Utilities in DGL-LifeSci for working with molecules.

RDKit Utils
-----------

RDKit utils for loading molecules and accessing their information.

.. autosummary::
    :toctree: ../generated/

    dgllife.utils.get_mol_3d_coordinates
    dgllife.utils.load_molecule
    dgllife.utils.multiprocess_load_molecules

Graph Construction
------------------

The modeling of graph neural networks starts with constructing appropriate graph topologies. We provide
three common graph constructions:

* ``bigraph``: Bi-directed graphs corresponding exactly to molecular graphs
* ``complete_graph``: Graphs with all pairs of atoms connected
* ``nearest_neighbor_graph``: Graphs where each atom is connected to its closest (k) atoms based on molecule coordinates

.. autosummary::
    :toctree: ../generated/

    dgllife.utils.mol_to_graph
    dgllife.utils.smiles_to_bigraph
    dgllife.utils.mol_to_bigraph
    dgllife.utils.smiles_to_complete_graph
    dgllife.utils.mol_to_complete_graph
    dgllife.utils.k_nearest_neighbors
    dgllife.utils.mol_to_nearest_neighbor_graph
    dgllife.utils.smiles_to_nearest_neighbor_graph

Featurization for Molecules
---------------------------

To apply graph neural networks, we need to prepare node and edge features for molecules. Intuitively,
they can be developed based on various descriptors (features) of atoms/bonds/molecules. Particularly, we can
work with numerical descriptors directly or use ``one_hot_encoding`` for categorical descriptors. When using
multiple descriptors together, we can simply concatenate them with ``ConcatFeaturizer``.

General Utils
```````````

.. autosummary::
    :toctree: ../generated/

    dgllife.utils.one_hot_encoding
    dgllife.utils.ConcatFeaturizer

Featurization for Nodes
```````````````````````

We consider the following atom descriptors:

* type/atomic number
* degree (excluding neighboring hydrogen atoms)
* total degree (including neighboring hydrogen atoms)
* explicit valence
* implicit valence
* hybridization
* total number of neighboring hydrogen atoms
* formal charge
* number of radical electrons
* aromatic atom
* ring membership
* chirality
* mass

We can employ their numerical values directly or with one-hot encoding.

.. autosummary::
    :toctree: ../generated/

    dgllife.utils.atom_type_one_hot
    dgllife.utils.atomic_number_one_hot
    dgllife.utils.atomic_number
    dgllife.utils.atom_degree_one_hot
    dgllife.utils.atom_degree
    dgllife.utils.atom_total_degree_one_hot
    dgllife.utils.atom_total_degree
    dgllife.utils.atom_explicit_valence_one_hot
    dgllife.utils.atom_explicit_valence
    dgllife.utils.atom_implicit_valence_one_hot
    dgllife.utils.atom_implicit_valence
    dgllife.utils.atom_hybridization_one_hot
    dgllife.utils.atom_total_num_H_one_hot
    dgllife.utils.atom_total_num_H
    dgllife.utils.atom_formal_charge_one_hot
    dgllife.utils.atom_formal_charge
    dgllife.utils.atom_num_radical_electrons_one_hot
    dgllife.utils.atom_num_radical_electrons
    dgllife.utils.atom_is_aromatic_one_hot
    dgllife.utils.atom_is_aromatic
    dgllife.utils.atom_is_in_ring_one_hot
    dgllife.utils.atom_is_in_ring
    dgllife.utils.atom_chiral_tag_one_hot
    dgllife.utils.atom_mass

For using featurization methods like above in creating node features:

.. autosummary::
    :toctree: ../generated/

    dgllife.utils.BaseAtomFeaturizer
    dgllife.utils.BaseAtomFeaturizer.feat_size
    dgllife.utils.CanonicalAtomFeaturizer
    dgllife.utils.CanonicalAtomFeaturizer.feat_size

Featurization for Edges
```````````````````````

We consider the following bond descriptors:

* type
* conjugated bond
* ring membership
* stereo configuration

.. autosummary::
    :toctree: ../generated/

    dgllife.utils.bond_type_one_hot
    dgllife.utils.bond_is_conjugated_one_hot
    dgllife.utils.bond_is_conjugated
    dgllife.utils.bond_is_in_ring_one_hot
    dgllife.utils.bond_is_in_ring
    dgllife.utils.bond_stereo_one_hot

For using featurization methods like above in creating edge features:

.. autosummary::
    :toctree: ../generated/

    dgllife.utils.BaseBondFeaturizer
    dgllife.utils.BaseBondFeaturizer.feat_size
    dgllife.utils.CanonicalBondFeaturizer
    dgllife.utils.CanonicalBondFeaturizer.feat_size
