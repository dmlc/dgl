import dgl.backend as F
import numpy as np
import os.path as osp
import torch

from collections import defaultdict
from dgl import DGLGraph
from dgllife.utils.featurizers import *
from dgllife.utils import mol_to_complete_graph, bond_type_one_hot
from functools import partial
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures

# Constants
atom_set = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'unknown']
chiral_set = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
			  Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]
hybridization_set = [Chem.rdchem.HybridizationType.SP,
					 Chem.rdchem.HybridizationType.SP2,
					 Chem.rdchem.HybridizationType.SP3]


def atom_type_one_hot(atom, allowable_set):
	"""One hot encoding for the type of an atom.
	Maps inputs not in the allowable set to the last element:'unknown'.
	Parameters
	----------
	atom : rdkit.Chem.rdchem.Atom
		RDKit atom instance.
	allowable_set : list of str
		Atom types to consider.
		Default: 'H', 'C', 'N', 'O', 'S', 'Cl', 'Br', 'F', 'P', 'I', 'unknown'.
		Size: 11
	Returns
	-------
	list
		List of boolean values where at most one value is True.
	"""
	x = atom.GetAtomicNum()
	if x not in allowable_set:
		x = allowable_set[-1]
	return list(map(lambda s: x == s, allowable_set))


def atom_partial_charge(atom):
	"""Partial charge of an atom calculated from the whole molecule.
	Parameters
	----------
	atom : rdkit.Chem.rdchem.Atom
		RDKit atom instance.
	Returns
	-------
	list
		List containing one int indicating calculated partial charge of an atom.
	"""
	return [float(atom.GetProp('_GasteigerCharge'))]


class AtomFeaturizer(BaseAtomFeaturizer):
	"""A default featurizer for atoms.
	The atom features include:
	* **One hot encoding of the atom type**. The supported atom types include
	  'H', 'C', 'N', 'O', 'S', 'Cl', 'Br', 'F', 'P', 'I', 'unknown'. size: 11.
	* **One hot encoding of the atom chirality**. The supported chirality types include
	  'R' and 'S'. size: 2.
	* **Formal charge of the atom**. size:1.
	* **Partial charge of the atom**. size: 1.
	* **One hot encoding of the atom hybridization**. The supported possibilities include
	  ``SP``, ``SP2``, ``SP3``. size: 3.
	* **Whether the atom is aromatic**. size: 1.
	* **One hot encoding of whether the atom is a hydrogen bond donor and/or acceptor**. size: 2.
	* **the number of rings that include this atom**. The supported ring size used include
	  3–8. size: 6.
	**We assume the resulting DGLGraph will not contain any virtual nodes.**
	Parameters
	----------
	atom_data_field : str
		Name for storing atom features in DGLGraphs, default to be 'h'.
	"""

	def __init__(self, atom_data_field='h'):
		super(AtomFeaturizer, self).__init__(
			featurizer_funcs={atom_data_field: ConcatFeaturizer(
				[partial(atom_type_one_hot, allowable_set=atom_set),
				 partial(atom_chiral_tag_one_hot, allowable_set=chiral_set, encode_unknown=False),
				 atom_formal_charge,
				 atom_partial_charge,
				 partial(atom_hybridization_one_hot, allowable_set=hybridization_set, encode_unknown=False),
				 atom_is_aromatic]
			)})

	def atom_is_H_bond_donor_or_acceptor(self, mol_feats):
		"""One hot encoding of whether the atom is a hydrogen bond donor and/or acceptor.
		Parameters
		----------
		mol_feats : tuple
			Features for molecules.
		Returns
		-------
		is_donor: defaultdict
			the atom id dict for H bond donor
		is_acceptor: : defaultdict
			the atom id dict for H bond acceptor
		"""
		is_donor = defaultdict(int)
		is_acceptor = defaultdict(int)
		# Get atom Ids including H bonds features
		for i in range(len(mol_feats)):
			if mol_feats[i].GetFamily() == 'Donor':
				node_list = mol_feats[i].GetAtomIds()
				for u in node_list:
					is_donor[u] = 1
			elif mol_feats[i].GetFamily() == 'Acceptor':
				node_list = mol_feats[i].GetAtomIds()
				for u in node_list:
					is_acceptor[u] = 1
		return is_donor, is_acceptor

	def __call__(self, mol):
		"""Featurize all atoms in a molecule.
		Parameters
		----------
		mol : rdkit.Chem.rdchem.Mol
			RDKit molecule instance.
		Returns
		-------
		dict
			For each function in self.featurizer_funcs with the key ``k``, store the computed
			feature under the key ``k``. Each feature is a tensor of dtype float32 and shape
			(N, M), where N is the number of atoms in the molecule.
		"""
		AllChem.ComputeGasteigerCharges(mol)
		num_atoms = mol.GetNumAtoms()
		atom_features = defaultdict(list)
		# get molecular features
		fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
		mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
		mol_feats = mol_featurizer.GetFeaturesForMol(mol)
		# get ring features
		sssr = Chem.GetSymmSSSR(mol)

		# Compute features for each atom
		is_donor, is_acceptor = self.atom_is_H_bond_donor_or_acceptor(mol_feats)

		for i in range(num_atoms):
			atom = mol.GetAtomWithIdx(i)
			for feat_name, feat_func in self.featurizer_funcs.items():
				atom_features[feat_name].append(feat_func(atom))
			# add H bonds features
			if is_donor[i]:
				atom_features[feat_name][i].append(1)
			else:
				atom_features[feat_name][i].append(0)
			if is_acceptor[i]:
				atom_features[feat_name][i].append(1)
			else:
				atom_features[feat_name][i].append(0)
			# add rings features
			count = [0] * 6
			for j in range(len(sssr)):
				if i in sssr[j] and 3 <= len(sssr[j]) <= 8:
					count[len(sssr[j])-3] += 1
			atom_features[feat_name][i] += count

		# Stack the features and convert them to float arrays
		processed_features = dict()
		for feat_name, feat_list in atom_features.items():
			feat = np.stack(feat_list)
			processed_features[feat_name] = F.zerocopy_from_numpy(feat.astype(np.float32))
		print(processed_features)
		return processed_features


def edge_featurizer(mol, max_distance):
	"""Featurize edges of a complete graph based on a molecule.

	Parameters
	----------
	mol : rdkit.Chem.rdchem.Mol
		RDKit molecule holder
	max_distance : int
		Maximum number to consider for the number of bonds between
		pairs of atoms.

	Returns
	-------
	dict : Mapping 'feat' to a float32 tensor of shape (V^2, D)
		Edge features. V is the number of atoms in the molecule and
		D for the feature size.
	"""
	# Part 1 based on number of bonds between each pair of atoms
	distance_matrix = torch.from_numpy(Chem.GetDistanceMatrix(mol))
	# Change shape from (V, V, 1) to (V^2, 1)
	distance_matrix = distance_matrix.float().reshape(-1, 1)
	# Elementwise compare if distance is bigger than 0, 1, ..., max_distance - 1
	distance_indicators = (distance_matrix >
						   torch.arange(0, max_distance).float()).float()

	# Part 2 for one hot encoding of bond type. There are 4 possible
	# bond types -- single, double, triple or aromatic
	num_atoms = mol.GetNumAtoms()
	bond_indicators = torch.zeros(num_atoms, num_atoms, 4)
	for bond in mol.GetBonds():
		bond_type_encoding = torch.tensor(bond_type_one_hot(bond)).float()
		begin_atom_idx, end_atom_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
		bond_indicators[begin_atom_idx, end_atom_idx] = bond_type_encoding
		bond_indicators[end_atom_idx, begin_atom_idx] = bond_type_encoding
	# Reshape from (V, V, 4) to (V^2, 4)
	bond_indicators = bond_indicators.reshape(-1, 4)

	# Part 3 for whether a pair of atoms belongs to a same ring.
	sssr = Chem.GetSymmSSSR(mol)
	ring_mate_indicators = torch.zeros(num_atoms, num_atoms, 1)
	for ring in sssr:
		ring = list(ring)
		num_atoms_in_ring = len(ring)
		for i in range(num_atoms_in_ring):
			ring_mate_indicators[i, torch.tensor(ring)] = 1
	ring_mate_indicators = ring_mate_indicators.reshape(-1, 1)

	return {'feat': torch.cat([distance_indicators, bond_indicators, ring_mate_indicators], dim=1)}


def construct_graph_and_featurize(mol, max_distance=7):
	"""Construct and featurize graphs from molecules as in Weave.

	Code adapted from Chainer Chemistry (https://github.com/chainer/
	chainer-chemistry/blob/da2507b38f903a8ee333e487d422ba6dcec49b05/
	chainer_chemistry/dataset/preprocessors/weavenet_preprocessor.py#L243)

	Parameters
	----------
	mol : rdkit.Chem.rdchem.Mol
		RDKit molecule holder

	Returns
	-------
	g : DGLGraph
		Constructed DGLGraph with edge features stored in g.edata['feat']
	"""
	num_atoms = mol.GetNumAtoms()
	g = mol_to_complete_graph(mol, add_self_loop=True,
							  node_featurizer=node_featurizer,
							  edge_featurizer=partial(edge_featurizer, max_distance=max_distance),
							  canonical_atom_order=False)

	return g


if __name__ == '__main__':
	# Take Aspirin for an example
	smiles = 'O(C(C([H])([H])[H])=O)C1=C([H])C([H])=C([H])C([H])=C1C(=O)O[H]'
	# smiles = 'OC1C2C1CC2'
	mol = Chem.MolFromSmiles(smiles)
	node_featurizer = AtomFeaturizer()
	node_featurizer(mol)
	g = construct_graph_and_featurize(mol)
	print(g)
	print(g.edata['feat'], g.ndata['h'])

