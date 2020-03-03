import dgl.backend as F
import numpy as np
from rdkit import Chem
import rdkit.Chem.EState as EState
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import rdkit.Chem.rdPartialCharges as rdPartialCharges
from collections import defaultdict
from dgl.data.chem import *
from functools import partial

import itertools
import numpy as np

from collections import defaultdict


def read_data(path):
    """Process data from a text file."""
    with open(path, 'r') as f:
        src, edits = [], []
        for line in f:
            smiles, label = line.strip("\t").split()
            src.append(smiles)
            edits.append(label)
    return src, edits


smiles_list, label_list = read_data("data/train.txt")

# Constants
atom_set = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'metal']  # How about atoms like Si?
chiral_set = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
              Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]  # R or S?
hybridization_set = [Chem.rdchem.HybridizationType.SP,
                     Chem.rdchem.HybridizationType.SP2,
                     Chem.rdchem.HybridizationType.SP3]
ATOM_FDIM = len(atom_set) + 2 + 1 + 1 + 6 + 3 + 2 + 1


def atom_formal_charge(atom):
    return [atom.GetFormalCharge()]


def atom_partial_charge(atom):
    return [float(atom.GetProp('_GasteigerCharge'))]
# AllChem.ComputeGasteigerCharges(smiles) used when loading smiles as a whole
# that's also what I mentioned as calculated electron density


def atom_in_ring_one_hot(atom):
    ring_list = []
    for i in range(3, 9):
        judge = atom.IsInRingSize(i)
        if judge:
            ring_list.append(1)
        else:
            ring_list.append(0)
    return ring_list


def atom_is_Hdonor_or_Hacceptor(atom):
    return 2


class AtomFeaturizer(BaseAtomFeaturizer):
    def __init__(self, atom_data_field='h'):
        super(AtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer(
                [partial(atom_type_one_hot, allowable_set=atom_set, encode_unknown=False),
                 partial(atom_chiral_tag_one_hot, allowable_set=chiral_set, encode_unknown=False),
                 atom_formal_charge,
                 atom_partial_charge,
                 atom_in_ring_one_hot,
                 partial(atom_hybridization_one_hot, allowable_set=hybridization_set, encode_unknown=False),
                 atom_is_Hdonor_or_Hacceptor,
                 atom_is_aromatic]
            )})


def graph_distance_one_hot(bond):
    return 7


class PairFeaturizer(BaseBondFeaturizer):
    def __init__(self, bond_data_field='e'):
        super(PairFeaturizer, self).__init__(
            featurizer_funcs={bond_data_field: ConcatFeaturizer(
                [bond_type_one_hot,
                 graph_distance_one_hot,
                 bond_is_in_ring_one_hot]
            )})
        