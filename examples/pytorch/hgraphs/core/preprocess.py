"""
Defines functions to turn a list of molecular SMILES strings into:
 1. A vocabulary of atom types, bond types, motifs, and
    attachment configurations.
 2. A list of hierarchical graphs representing each molecule.
 3. A list of hierarchical graphs representing each motif in the vocabulary.

Code is logically organized from bottom to top.

Abbreviations:
hgraph := hierarchical graph
attch := attachment atom
frag := fragment of molecule
orig := original molecule (before fragmentation)
"""

from rdkit import Chem
from collections import Counter, namedtuple, defaultdict
from pathlib import Path
from tqdm import tqdm
import json, networkx as nx, dgl, torch

"""
Represents a pair of atom instances at which two motifs attach
in a fragmented mol.

Args:
parent - Index of the instance of the attch in the parent motif.
child - Index of the instance of the attch in the child motif.
copy_in - "parent" or child", identifies which motif the
    original attch instance and copy instance is in.
"""
AttachmentPair = namedtuple("AttachmentPair", ["parent", "child", "copy_in"])

def detach_motifs(fragmented_mol, attch_pairs, atom,
                  atom_idx, neighbor, neighbor_idx, bond_order,
                  decomp_motif_atom_idxs):
    """
    Remove bond between attch and anchor,
    create copy of attch, and bond anchor to the copy.
    Now mol has been fragmented into 2 motifs and can be
    unfragmented by "attaching" motifs at the attchs in each motif,
    i.e. merging attch in original motif and its copy in child motif.
    """

    #If either has no other neighbors, make it the anchor,
    #so its motif will be the bridge (won't be lone atom).
    atom_has_other_nbrs = atom.GetDegree() > 1
    if atom_has_other_nbrs:
        anchor, anchor_idx = neighbor, neighbor_idx
        attch, attch_idx = atom, atom_idx
    else:
        anchor, anchor_idx = atom, atom_idx
        attch, attch_idx = neighbor, neighbor_idx

    fragmented_mol.RemoveBond(attch_idx, anchor_idx)
    attch_copy = Chem.Atom(attch.GetAtomicNum())
    attch_copy.SetFormalCharge(attch.GetFormalCharge())
    #The atom object created in the molecule is different from the isolated one.
    attch_copy_idx = fragmented_mol.AddAtom(attch_copy)
    attch_copy = fragmented_mol.GetAtomWithIdx(attch_copy_idx)
    fragmented_mol.AddBond(anchor_idx, attch_copy_idx, bond_order)
  
    #The anchor is the one that gets the copy attch and bridge;
    #if it's already the child of another motif, that means the entire frag
    #it's being split off is the child of that motif, so the frag
    #with the original attch must be the child of the frag with the copy attch.
    #Or, if attch is atom's single nbr, parent motif will be the one with the
    #atom and nbr's copy.
    if anchor.HasProp("is_child_of_motif") or not atom_has_other_nbrs:
        parent_idx, child_idx, copy_in = attch_copy_idx, attch_idx, "parent"
        attch.SetBoolProp("is_child_of_motif", True)
    #If attch is atom, parent motif will be the one the copy is detached from. 
    elif atom_has_other_nbrs:
        parent_idx, child_idx, copy_in = attch_idx, attch_copy_idx, "child"
        attch_copy.SetBoolProp("is_child_of_motif", True)

    attch_pairs.append(AttachmentPair(parent = parent_idx, child = child_idx, copy_in = copy_in))
    
    #The attachment copy may itself form the endpoint of a fragment on the child motif
    #and will be eligible for decomposition.
    if decomp_motif_atom_idxs is not None and attch_idx in decomp_motif_atom_idxs:
        decomp_motif_atom_idxs.append(attch_copy_idx)

    #For debugging: Make attch parent and child have the same label when drawing the molecule.
    #attch_copy.SetAtomMapNum(attch_idx)

def update_atom_graph(hgraph, atom, atom_idx, neighbor_indices, bond_orders):
    """
    Add an atom and its edges to its neighbors to the hgraph.

    Args:
    bond_orders - ...[i] := RDKit BondType for ith edge of the atom.
    """
    
    #Don't need the typical add_edges([in, out], [out, in]) pattern for bidirected edges since this func will eventually
    #be called on each atom in each bond separately.
    hgraph.nodes["atom"].data["features"][atom_idx] = torch.tensor(
        [atom.GetAtomicNum(), atom.GetFormalCharge()]
    )
    source_indices = [atom_idx]
    target_indices = neighbor_indices
    edge_data = { "bond_order": torch.tensor(bond_orders) } #Bond order obs get converted to nums.
    hgraph.add_edges(source_indices, target_indices,
                     edge_data, etype = ("atom", "bond", "atom"))

def find_and_detach_motifs(
    step, mol, fragmented_mol, attch_pairs, hgraph,
    atom, atom_idx, visited, discovered_at_step,
    earliest_discovered_descendent,
    decomp_motif_atom_idxs, parent_idx = -1):
    """
    Get the edges of an atom in the mol, add them to the hgraph,
    check whether the atom is an attch between motifs, and
    if so record it and detach the motifs in fragmented_mol.
    Recursively do the same for the atom's unvisited neighbors.
    
    Args:
    step - Singleton int: [step_of_search].
    mol - The original RDKit Mol object.
    fragmented_mol - RDKit Mol object with motifs in the process
        of being separated into fragments.
    attch_pairs - List of AttachmentPairs.
    atom - The atom being processed.
    atom_idx - The idx of the atom being processed in the mol.
    decomp_motif_atom_idxs - List of atom idxs of motifs to decompose into bonds/rings or None.
    parent_idx - Idx of neighbor of atom that called this func on it
        (-1 if atom is root).
    """
    
    visited[atom_idx] = True
    discovered_at_step[atom_idx] = step[0]
    earliest_discovered_descendent[atom_idx] = step[0]
    step[0] += 1
    neighbor_indices = []
    bond_orders = []
    for neighbor in atom.GetNeighbors():
        neighbor_idx = neighbor.GetIdx()
        neighbor_indices.append(neighbor_idx)
        bond_order = fragmented_mol.GetBondBetweenAtoms(
            atom_idx, neighbor_idx
        ).GetBondType()
        bond_orders.append(bond_order)
        
        if not visited[neighbor_idx]:
            find_and_detach_motifs(
                step, mol, fragmented_mol, attch_pairs,
                hgraph, neighbor, neighbor_idx, visited,
                discovered_at_step, earliest_discovered_descendent,
                decomp_motif_atom_idxs, parent_idx = atom_idx
            )
            earliest_discovered_descendent[atom_idx] = (
                min(earliest_discovered_descendent[atom_idx],
                    earliest_discovered_descendent[neighbor_idx])
            )
            #Is it a bridge bond?
            #(no descendants of this atom are connected to atoms in the graph explored before it?)
            if (earliest_discovered_descendent[neighbor_idx] >
                discovered_at_step[atom_idx]):
                atom_in_ring = atom.IsInRing()
                neighbor_in_ring = neighbor.IsInRing()
                either_in_ring = (atom_in_ring or neighbor_in_ring)
                both_have_nbrs = (len(atom.GetBonds()) >= 2 and
                                        len(neighbor.GetBonds()) >= 2)
                either_has_nbrs = (len(atom.GetBonds()) >= 2 or
                                        len(neighbor.GetBonds()) >= 2)
                is_valid_bridge = (both_have_nbrs and
                                   either_in_ring)
                both_in_decomp_motifs = (decomp_motif_atom_idxs is not None and
                                         (atom_idx in decomp_motif_atom_idxs and
                                         neighbor_idx in decomp_motif_atom_idxs)
                                        )
                can_decompose_motifs = both_in_decomp_motifs and either_has_nbrs
                
                #Is it a chemically valid bridge bond?
                if is_valid_bridge or can_decompose_motifs:
                    detach_motifs(
                        fragmented_mol, attch_pairs,
                        atom, atom_idx,
                        neighbor, neighbor_idx, bond_order,
                        decomp_motif_atom_idxs
                    )
        elif parent_idx != neighbor_idx:
            earliest_discovered_descendent[atom_idx] = min(
                earliest_discovered_descendent[atom_idx],
                discovered_at_step[neighbor_idx]
            )
   
    #Only build atom graphs in the 1st step of build_vocabs, before
    #decomposing motifs. Otherwise atoms will get duplicated.
    if decomp_motif_atom_idxs is None:
        update_atom_graph(hgraph, atom, atom_idx, neighbor_indices,
                      bond_orders)

def get_motif_ids_of_atoms(atom_ids_of_motifs, num_atoms):
    """
    Args:
    atom_ids_motif - list[i] := list of atom ids of motif i
    """

    motif_ids_of_atoms = [None] * num_atoms

    for motif_idx, motif in enumerate(atom_ids_of_motifs):
        for atom_idx in motif:
            motif_ids_of_atoms[atom_idx] = motif_idx

    return motif_ids_of_atoms

def root_motifs(motifs, atom_ids_of_motifs, motif_ids_of_atoms):
    """
    Make sure root motif will be the first one, i.e. idx 0 in the graph,
    so that it's the root of the traversal when training the generator.
    """
   
    root_motif_old_id = motif_ids_of_atoms[0]
    
    root_motif = motifs[root_motif_old_id]
    first_motif = motifs[0]
    
    motifs[0] = root_motif
    motifs[root_motif_old_id] = first_motif
    
    root_motif_atom_ids = atom_ids_of_motifs[root_motif_old_id]
    first_motif_atom_ids = atom_ids_of_motifs[0]
    
    atom_ids_of_motifs[0] = root_motif_atom_ids
    atom_ids_of_motifs[root_motif_old_id] = first_motif_atom_ids

    for i, motif_id in enumerate(motif_ids_of_atoms):
        if motif_id == 0:
            motif_ids_of_atoms[i] = root_motif_old_id
        elif motif_id == root_motif_old_id:
            motif_ids_of_atoms[i] = 0

def build_motif_and_attch_conf_trees_multiple(
    fragmented_mol, num_atoms_original,
    num_atoms_after_frag, attch_pairs, hgraph, metadata_only
    ):
    """
    Build the tree when there's more than 1 motif in the molecule, i.e. there are attachments.
    """
    
    motifs = list(Chem.rdmolops.GetMolFrags(fragmented_mol, asMols = True, sanitizeFrags = True))
    atom_ids_of_motifs = list(Chem.rdmolops.GetMolFrags(fragmented_mol))
    motif_ids_of_atoms = get_motif_ids_of_atoms(atom_ids_of_motifs, num_atoms_after_frag)

    #Idxs of atoms in frags generated by GetMolFrags() depend on the mol and
    #are different from the "canonical" idxs RDK would give them if we did MolToSmiles(frag SMILES)
    #so atoms will not appear in the same order in GetMolFrags() depending on where the motif is.
    #_smilesAtomOutputOrder will give us a map (atom canon idx) -> (atom instance idx).
    motif_atom_ids_of_motifs = []
    for motif in motifs:
        Chem.MolToSmiles(motif)
        motif_atom_ids_of_motifs.append(list(motif.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"]))

    attch_configs = [defaultdict(list) for _ in range(len(motifs))]
    
    if not metadata_only:
        root_motifs(motifs, atom_ids_of_motifs, motif_ids_of_atoms)
        smiles = [Chem.MolToSmiles(motif) for motif in motifs]
       
        #IDs of attachment atom copies will get replaced with IDs of the original attachment atoms.
        original_atom_ids_of_motifs = [list(atom_ids_of_motif) for atom_ids_of_motif in atom_ids_of_motifs]
        
        graph_src_indices = []
        graph_dst_indices = []
        attch_motif_idx_pairs = [] #[i] := [parent atom idx in its motif, child atom idx in its motif]
        
        for attch_pair in attch_pairs:
            parent_motif_idx = motif_ids_of_atoms[attch_pair.parent]
            child_motif_idx = motif_ids_of_atoms[attch_pair.child]
            graph_src_indices.append(parent_motif_idx)
            graph_dst_indices.append(child_motif_idx)
            
            parent_atom_idx_in_motif = atom_ids_of_motifs[parent_motif_idx].index(attch_pair.parent)
            parent_atom_idx_in_canon_motif = motif_atom_ids_of_motifs[parent_motif_idx].index(parent_atom_idx_in_motif)
            child_atom_idx_in_motif = atom_ids_of_motifs[child_motif_idx].index(attch_pair.child)
            child_atom_idx_in_canon_motif = motif_atom_ids_of_motifs[child_motif_idx].index(child_atom_idx_in_motif)
            attch_motif_idx_pair = [parent_atom_idx_in_canon_motif, child_atom_idx_in_canon_motif]
            attch_motif_idx_pairs.append(attch_motif_idx_pair)
            
            attch_type = fragmented_mol.GetAtomWithIdx(attch_pair.parent).GetAtomicNum()
            attch_configs[parent_motif_idx][attch_type].append(parent_atom_idx_in_canon_motif) 
            attch_configs[child_motif_idx][attch_type].append(child_atom_idx_in_canon_motif)
         
            if attch_pair.copy_in == "child":
                original_atom_ids_of_motifs[child_motif_idx].remove(attch_pair.child)
                original_atom_ids_of_motifs[child_motif_idx].append(attch_pair.parent)
            elif attch_pair.copy_in == "parent":
                original_atom_ids_of_motifs[parent_motif_idx].remove(attch_pair.parent)
                original_atom_ids_of_motifs[parent_motif_idx].append(attch_pair.child)

        #Add intralevel edges.
        attachment_data = {
            "attachment_motif_id_pair": torch.tensor(attch_motif_idx_pairs), 
        }
        hgraph.add_edges(graph_src_indices, graph_dst_indices,
                         attachment_data,
                         etype = ("motif", "attaches to", "motif"))
        #The attch configs are connected the same way their corresponding motifs are.
        hgraph.add_edges(graph_src_indices, graph_dst_indices,
                         attachment_data,
                         etype = ("attachment_config", "attaches to", "attachment_config"))
        
        #Add interlevel edges.
        #Attachment atoms will be part of 2 motifs.
        atom_to_attch_conf_srcs = []
        atom_to_attch_conf_dsts = []
        for motif_i, motif_orig_atom_ids in enumerate(original_atom_ids_of_motifs):
            atom_to_attch_conf_srcs.extend(motif_orig_atom_ids)
            atom_to_attch_conf_dsts.extend(
                [motif_i] * len(motif_orig_atom_ids)
            )
        hgraph.add_edges(atom_to_attch_conf_srcs,
                         atom_to_attch_conf_dsts,
                         etype = ("atom", "of", "attachment_config"))
        hgraph.add_edges(hgraph.nodes("attachment_config"),
                         hgraph.nodes("motif"),
                         etype = ("attachment_config", "of", "motif"))

    return atom_ids_of_motifs, motifs, attch_configs, hgraph
    
def build_motif_and_attch_conf_trees_single(fragmented_mol, num_atoms_original, hgraph, metadata_only):
    """
    Build the tree when there's only a single motif in the molecule, i.e. no attachments.
    """

    motifs = Chem.rdmolops.GetMolFrags(fragmented_mol, asMols = True, sanitizeFrags = False)
    atom_ids_of_motifs = [list(range(num_atoms_original))]
    original_atom_ids_of_motifs = atom_ids_of_motifs
    motif_ids_of_original_atoms = [0] * num_atoms_original
    attch_configs = [defaultdict(list)] 

    if not metadata_only:
        #Add node at each level.
        hgraph.add_nodes(1, ntype = "motif")
        hgraph.add_nodes(1, ntype = "attachment_config")
        
        #Add interlevel edges.
        hgraph.add_edges(hgraph.nodes("atom"), torch.tensor(motif_ids_of_original_atoms),
                         etype = ("atom", "of", "attachment_config"))
        hgraph.add_edges(hgraph.nodes("attachment_config"), hgraph.nodes("motif"),
                         etype = ("attachment_config", "of", "motif"))

    return atom_ids_of_motifs, motifs, attch_configs, hgraph

def build_motif_and_attch_conf_trees(fragmented_mol, num_atoms_original,
                                     num_atoms_after_frag, attch_pairs, hgraph,
                                     metadata_only):
    """
    Use the AttachmentPairs recorded during fragmentation to build
    DGLGraphs of the motifs and attch configs into the hgraph.

    Args:
    num_atoms_original - Num atoms in provided mol prior to fragmentation.
    num_atoms_after_frag - Num atoms including additional copies
        of attchs created after fragmentation.
    metadata_only - Whether to actually build the graph or just get metadata used for building it.

    Returns:
    atom_ids_of_motifs - 
        ...[i] := list of motif i's atom ids in fragmented mol.
    motifs - 
        ...[i] := RDKit Mol object representing motif i.
    attch_configs - 
                   ...[i] := defaultdict 
        defaultdict[k][j] := jth atom of atomic num k in
                             motif i's attch config.
    hgraph - 3-level DGL graph describing the mol's
        atoms, attch configs, and motifs, with the
        relations specified in get_empty_hierarchical_graph().
    """

    if len(attch_pairs) > 0:
        atom_ids_of_motifs, motifs, attch_configs, hgraph = build_motif_and_attch_conf_trees_multiple(
            fragmented_mol, num_atoms_original, num_atoms_after_frag,
            attch_pairs, hgraph, metadata_only
        )
    else:
        assert num_atoms_original == num_atoms_after_frag
        atom_ids_of_motifs, motifs, attch_configs, hgraph = build_motif_and_attch_conf_trees_single(
            fragmented_mol, num_atoms_original, hgraph, metadata_only
        )

    return atom_ids_of_motifs, motifs, attch_configs, hgraph

def get_empty_hierarchical_graph():
    empty_hgraph = dgl.heterograph({
            ("atom", "bond", "atom"): ([], []),
            ("atom", "of", "attachment_config"): ([], []),
            ("attachment_config", "attaches to", "attachment_config"): ([], []),
            ("attachment_config", "ancestry", "attachment_config"): ([], []),
            ("attachment_config", "of", "motif"): ([], []),
            ("motif", "attaches to", "motif"): ([], []),
            ("motif", "ancestry", "motif"): ([], [])
    })

    return empty_hgraph

def fragment_mol(mol, num_atoms, hgraph, attch_pairs, decomp_motif_atom_idxs):
    """
    Fragment a molecule into separate motifs and get a list of the pairs of atoms
    at which they attach.

    Args:
    mol - RDKit Mol
    decomp_motif_atom_idxs - List of atom idxs of motifs to decompose into bonds/rings.
    """

    fragmented_mol = Chem.RWMol(mol)
    
    #Non-kekule atomaticity imposes valence assumptions on ALL
    #elements of aromatic rings, which will be violated after
    #cutting their bonds during fragmentation.
    #fragmented_mol will get fed in as mol when decomposing motifs,
    #in step 2 of build_vocabs, so its bonds will have already been
    #cut and may not be kekulizable.
    if decomp_motif_atom_idxs is None: 
        Chem.Kekulize(fragmented_mol)

    atoms = mol.GetAtoms()
    step = [0] #Singleton representing # steps of the search through the atom graph.
    visited = [False] * num_atoms
    discovered_at_step = [-1] * num_atoms
    earliest_discovered_descendent = [-1] * num_atoms #Values are the steps at which these descendents were discovered.

    for atom in atoms:
        atom_idx = atom.GetIdx()
        atom = fragmented_mol.GetAtomWithIdx(atom_idx)
        if not visited[atom_idx]:
            find_and_detach_motifs(
                step, mol, fragmented_mol, attch_pairs, hgraph,
                atom, atom_idx, visited, discovered_at_step,
                earliest_discovered_descendent, decomp_motif_atom_idxs)

    return mol, fragmented_mol, attch_pairs

def get_hierarchical_graph(mol, hgraph = None, attch_pairs = None, decomp_motif_atom_idxs = None):
    """
    Get a hierarchical graph of atoms, motifs, and attachment configurations
    from a molecule, as well as associated metadata.

    Args:
    mol - RDKit Mol.
    hgraph - An existing hgraph to build on.
    attch_pairs - List of AttachmentPairs if mol is already fragmented.
    decomp_motif_atom_idxs - List of atom idxs of motifs to decompose into bonds/rings.
    """ 

    num_atoms = len(mol.GetAtoms())
    
    if hgraph is None:
        hgraph = get_empty_hierarchical_graph()
        hgraph.add_nodes(num_atoms, ntype = "atom")
        hgraph.nodes["atom"].data["features"] = torch.empty(num_atoms, 2)
    
    attch_pairs = attch_pairs if attch_pairs is not None else []
    mol, fragmented_mol, attch_pairs = fragment_mol(mol, num_atoms, hgraph, attch_pairs, decomp_motif_atom_idxs)
    num_atoms_after_frag = len(fragmented_mol.GetAtoms()) #Attachment atoms get copied to both motifs after fragmentation.
    
    metadata_only = decomp_motif_atom_idxs is None
    (atom_ids_of_motifs,
     motifs, attch_configs, hgraph) = build_motif_and_attch_conf_trees(
        fragmented_mol, num_atoms, num_atoms_after_frag,
        attch_pairs, hgraph, metadata_only
    )

    return (mol, fragmented_mol, attch_pairs, atom_ids_of_motifs,
            motifs, attch_configs, hgraph)

def get_attch_config_words(attch_configs, motif_words, motif_vocab):
    """
    Returns:
    attch_config_words - 
        ...[i]   := set (of motif i's attch configs)
        e in set := tuple (of attchs of each atomic num)
        tuple[j] := (k, tuple) (k is an atomic num)
        tuple[n] := the index of the nth attch atom with atomic num k
                    in attch config e of the ith motif in the vocab.
    """

    attch_config_words = defaultdict(set)
    for i in range(len(attch_configs)):
        attch_config_i = attch_configs[i].copy()
        motif_vocab_idx = motif_vocab.index(motif_words[i])
        
        #Turn lists to tuples so attch configs can be put into a Set().
        for atom_type in attch_configs[i]:
            attch_config_i[atom_type] = tuple(attch_config_i[atom_type])
        
        attch_config_hashable = tuple(attch_config_i.items())
        attch_config_words[motif_vocab_idx].add(attch_config_hashable)

    return attch_config_words

def assert_attch_indices_below_num_atoms_in_motif(attch_config_words, motif_vocab):
    for motif_idx, attch_confs in attch_config_words.items():
        for attch_conf in attch_confs:
            for attch_indices in dict(attch_conf).values():
                num_atoms_in_motif = Chem.MolFromSmarts(motif_vocab[motif_idx]).GetNumAtoms()
                assert max(attch_indices) < num_atoms_in_motif

def format_attch_config_vocab(attch_config_vocab):
    """
    Change the attch config vocab from a dict of sets of defaultdicts with tuple values
    to a dict of lists of dicts with list values.
    """

    max_num_attch_configs = 0
    for motif_idx in attch_config_vocab.keys():
        attch_configs = attch_config_vocab[motif_idx]
        num_attch_configs = len(attch_configs)
        if num_attch_configs > max_num_attch_configs:
            max_num_attch_configs = num_attch_configs
        #So each of the motif's attch configs has a unique vocab idx.
        attch_config_vocab[motif_idx] = list(attch_config_vocab[motif_idx])
        
        for i in range(len(attch_configs)):
            #DefaulFormatting adict won't error on non-existing keys, so make it a dict.
            attch_config_vocab[motif_idx][i] = dict(attch_config_vocab[motif_idx][i])
            
            #Configs will be used to record unused attchs of motifs during generation,
            #so they should be mutable.
            for atom_type in attch_config_vocab[motif_idx][i]:
                attch_config_vocab[motif_idx][i][atom_type] = list(attch_config_vocab[motif_idx][i][atom_type])

    return dict(attch_config_vocab), max_num_attch_configs

def build_vocabs_from_mol(
    mol_idx, mol_SMILES, atom_vocab, bond_vocab,
    motif_counter, max_num_atoms_in_motif, attch_config_vocab,
    init_hgraphs_data, final_hgraphs_data, atom_ids_of_motifs, mols_motifs,
    mols_attachment_pairs, fragmented_mols,
    motifs_to_decomp = None
    ):
    """
    Incorporate this mol's atoms, motif, and attachment configs
    into the vocab.

    Args:
    motifs_to_decomp - List of motifs (SMILES) to decompose
    into rings/bonds.  
    """

    if "." in mol_SMILES:
        raise ValueError("This SMILES contains a '.', suggesting multiple molecules -- the input must be 1 molecule!")
    
    if motifs_to_decomp is not None:
        #Formatting already-extracted info needed for step 2 of build_vocabs.
        mol_uncommon_motif_ids = [motif_id for motif_id, motif in enumerate(mols_motifs[mol_idx])
                                 if motif in motifs_to_decomp]
        atom_ids_of_uncommon_motifs = [atom_id for motif_id in mol_uncommon_motif_ids
                                       for atom_id in atom_ids_of_motifs[mol_idx][motif_id]]
        mol = fragmented_mols[mol_idx]
        mol_attachment_pairs = mols_attachment_pairs[mol_idx]
        hgraph = init_hgraphs_data[mol_idx][6]
    else:
        mol = Chem.MolFromSmiles(mol_SMILES)
        atom_ids_of_uncommon_motifs = None
        mol_attachment_pairs = None
        hgraph = None

    hgraph_data = get_hierarchical_graph(mol, hgraph, mol_attachment_pairs, atom_ids_of_uncommon_motifs)
    (mol, fragmented_mol, mol_attachment_pairs, atom_ids_of_motifs_,
     motifs,
     attch_configs, hgraph) = hgraph_data
    
    #Only do in step 1 of build_vocabs, so info is avail for step 2.
    if motifs_to_decomp is None:
        atom_ids_of_motifs.append(atom_ids_of_motifs_)
        mols_motifs.append([Chem.MolToSmiles(motif) for motif in motifs])
        fragmented_mols.append(fragmented_mol)
        mols_attachment_pairs.append(mol_attachment_pairs)
        init_hgraphs_data.append(hgraph_data)
    else:
        final_hgraphs_data.append(hgraph_data)

    #Mol will have different motifs before and after decomposing uncommon motifs.
    motif_words = []
    for motif in motifs:
        #Hashable (but immutable) representations of the motifs, ions, and attch configs in this molecule.
        motif_words.append(Chem.MolToSmiles(motif))

        motif_num_atoms = motif.GetNumAtoms()
        if motif_num_atoms > max_num_atoms_in_motif[0]:
            max_num_atoms_in_motif[0] = motif_num_atoms
    motif_counter.update(motif_words)
    motif_vocab = list(motif_counter)
   
    #Only need to do this in step 2, after decomposing uncommon motifs.
    if motifs_to_decomp is not None:
        attch_config_words = get_attch_config_words(
            attch_configs, motif_words, motif_vocab
        )
        for motif_idx in attch_config_words.keys(): 
            attch_config_vocab[motif_idx].update(attch_config_words[motif_idx])
        
        atom_words = ((atom_num.item(), formal_charge.item()) for
                      atom_num, formal_charge in
                      hgraph.nodes["atom"].data["features"])
        atom_vocab.update(atom_words)

        assert_attch_indices_below_num_atoms_in_motif(attch_config_words, motif_vocab)
    
def build_vocabs(mols_SMILES, motif_min_occurences):
    """
    Build vocabs of motifs, atoms, and attachment configs. Done in 2 steps:
    1. Decompose mols into motifs and build motif vocab.
    2. Decompose uncommon motifs into rings/bonds and build all vocabs.

    Args:
    mols_SMILES - List of SMILES strings for mols.
    motif_min_occurences - The min occurences of a motif in the dataset for it
        not to be decomposed.

    Returns:
    hgraphs_data - List of outputs of get_hierarchical_graph for each SMILES string.
    atom_vocab - List of tuples (atomic number, formal charge).
    bond_vocab - List of integers representing bond types.
    motif_vocab - List of SMILES strings representing motifs.
    motif_counter - Counter of SMILES strings representing motifs.
    attch_config_vocab - A dict of attachment configs for each motif.
        dict[i] := list (of attachment configs for motif i)
        list[j] := dict (attachment config)
        dict[k] := list (atoms of type k)
        list[n] := nth atom of type k in the jth attachment config of motif i.
    max_num_attch_configs - Max num of distinct attch configs found for any single motif.
    max_num_atoms_in_motif - Max num of atoms in any motif.
    """

    atom_vocab = set()
    bond_vocab = list(Chem.rdchem.BondType.values) #1:Single, 2:double, 3:triple, 4:quadruple, 12:aromatic.
    motif_counter = Counter()
    max_num_atoms_in_motif = [0]
    attch_config_vocab = defaultdict(set)
    init_hgraphs_data = []
    final_hgraphs_data = []

    #Used to decompose uncommon motifs in each mol in step 2.
    mols_motifs = [] #[i] := list of motifs in mol i.
    atom_ids_of_motifs = [] #[i][j] := list of atom idxs of motif j in mol i.
    mols_attachment_pairs = [] #[i] := list of AttachmentPairs of atoms in mol i.
    fragmented_mols = [] #[i] := mol i fragmented into initial motifs.

    #Make vocabs of each motif, atom, and attch config.
    for mol_i, mol_SMILES in enumerate(tqdm(mols_SMILES)):
        build_vocabs_from_mol(
            mol_i, mol_SMILES, atom_vocab, bond_vocab,
            motif_counter, max_num_atoms_in_motif,
            attch_config_vocab, init_hgraphs_data, final_hgraphs_data,
            atom_ids_of_motifs, mols_motifs,
            mols_attachment_pairs, fragmented_mols
        )
 
    #Record uncommon motifs, i.e. they occur below the min_motif_occurences.
    motifs_to_decompose = []
    for motif_vocab_idx, (motif, count) in enumerate(motif_counter.items()):
        if count < motif_min_occurences:
            motifs_to_decompose.append(motif)
 
    print("Decomposing...")
    #Make vocabs, but also decompose uncommon motifs (so the motif vocab will change).
    motif_counter = Counter()
    max_num_atoms_in_motif = [0]
    for mol_i, mol_SMILES in enumerate(tqdm(mols_SMILES)):
        build_vocabs_from_mol(
            mol_i, mol_SMILES, atom_vocab, bond_vocab,
            motif_counter, max_num_atoms_in_motif,
            attch_config_vocab, init_hgraphs_data, final_hgraphs_data,
            atom_ids_of_motifs, mols_motifs,
            mols_attachment_pairs, fragmented_mols,
            motifs_to_decompose
    )
   
    max_num_atoms_in_motif = max_num_atoms_in_motif[0]
    attch_config_vocab, max_num_attch_configs = format_attch_config_vocab(attch_config_vocab)
    
    #Order the node vocab so they have unique indices.
    atom_vocab = list(atom_vocab)
    motif_vocab = list(motif_counter)

    return (final_hgraphs_data, atom_vocab, bond_vocab, motif_vocab,
            motif_counter, attch_config_vocab,
            max_num_attch_configs, max_num_atoms_in_motif)

def add_ancestry_connections(hgraph):
    """
    Add an edge from every u to v in the motif and attachment config graphs
    where u is an ancestor or descendent of v.
    Add a feature to each edge: 0 if v is u's descendent, n if v is u's nth ancestor.
    (so information gets propagated from child to ancestors depending on their distance,
     all the way upto the root motif, which will determine the graph's latent rep.)
    """

    num_nodes = hgraph.number_of_nodes("motif")
    num_tree_edges = hgraph.number_of_edges(("motif", "attaches to", "motif"))
    assert num_nodes == hgraph.number_of_nodes("attachment_config")
    assert num_tree_edges == hgraph.number_of_edges(("attachment_config", "attaches to", "attachment_config"))

    src_node_ids = []
    dst_node_ids = []
    ancestry_labels = []
    ancestors = [[] for node in range(num_nodes)]
    stack = []
    stack.append(0)
    while len(stack) != 0:
        node_id = stack.pop()

        _, child_ids, edge_ids = hgraph.out_edges(
            node_id,
            etype = ("motif", "attaches to", "motif"),
            form = "all"
        )
        for child_id in child_ids.tolist():
            stack.append(child_id)
            ancestors[child_id].append(node_id)
            ancestors[child_id].extend(ancestors[node_id])
            num_ancestors = len(ancestors[child_id])

            #Child -> Parents
            src_node_ids.extend([child_id] * num_ancestors)
            dst_node_ids.extend(ancestors[child_id])
            ancestry_labels.extend(range(1, num_ancestors + 1))
            
            #Parents -> Child
            src_node_ids.extend(ancestors[child_id])
            dst_node_ids.extend([child_id] * num_ancestors)
            ancestry_labels.extend([0] * num_ancestors)

    #We can treat an ancestry label (0, 1, ..., n) as its own vocab index
    #in an implicit vocabulary of ancestry labels.
    data = { "vocab_idx": torch.tensor(ancestry_labels) }
    
    #Add new ancestry edges.
    hgraph.add_edges(src_node_ids, dst_node_ids, data,
                     etype = ("motif", "ancestry", "motif"))
    hgraph.add_edges(src_node_ids, dst_node_ids, data,
                     etype = ("attachment_config", "ancestry", "attachment_config"))

    max_path_length = max(ancestry_labels) if len(ancestry_labels) > 0 else 0

    return max_path_length

def label_motif_graph(hgraph, motif_vocab, motifs):
    """
    Label the nodes of the motif-level graph of a molecular hgraph
    with their type's index in the vocabulary. Not "motif graph"
    as in a graph of atoms representing a single motif.
    """

    motif_words = [Chem.MolToSmiles(motif) for motif in motifs]
    node_words_vocab_indices = [motif_vocab.index(word) for word in motif_words]
    hgraph.nodes["motif"].data["vocab_idx"] = torch.tensor(node_words_vocab_indices)

    return motif_words

def label_attachment_config_graph(hgraph, attch_config_vocab, attch_configs, motif_vocab, motif_words): 
    attch_config_words = [
       (motif_vocab.index(motif_words[i]), tuple(attch_configs[i].items())) 
        for i in range(len(attch_configs))
    ]
    
    node_words_vocab_indices = [
        attch_config_vocab[motif_idx].index(dict(attch_config))
        for motif_idx, attch_config
        in attch_config_words
    ]
    hgraph.nodes["attachment_config"].data["vocab_idx"] = torch.tensor(node_words_vocab_indices)

def label_atom_graph(hgraph, atom_vocab, bond_vocab):
    atom_words = [
        (atom_num.item(), formal_charge.item())
        for atom_num, formal_charge
        in hgraph.nodes["atom"].data["features"]
    ]
    
    node_words_vocab_indices = [atom_vocab.index(word) for word in atom_words]
    hgraph.nodes["atom"].data["vocab_idx"] = torch.tensor(node_words_vocab_indices)
    edge_words_vocab_indices = [
        bond_vocab.index(bond_order)
        for bond_order
        in hgraph.edges[("atom", "bond", "atom")].data["bond_order"]
    ]
    hgraph.edges[("atom", "bond", "atom")].data["vocab_idx"] = (
        torch.tensor(edge_words_vocab_indices)
    )

def get_final_graph(hgraph_data, max_path_length, motif_SMILES_to_graph, fragmented_mol_SMILES,
                    mol_hgraphs, atom_vocab, bond_vocab, motif_vocab, attch_config_vocab):
    (mol, fragmented_mol, attch_pairs, atom_ids_of_motifs,
     motifs, attch_configs, hgraph) = hgraph_data
    fragmented_mol_SMILES.append(Chem.MolToSmiles(fragmented_mol))

    motif_words = label_motif_graph(hgraph, motif_vocab, motifs)
    label_attachment_config_graph(hgraph, attch_config_vocab, attch_configs, motif_vocab, motif_words) 
    label_atom_graph(hgraph, atom_vocab, bond_vocab)
    
    max_path_length[0] = max(max_path_length[0], add_ancestry_connections(hgraph))

    mol_hgraphs.append(hgraph)

def mol_to_atom_graph(mol, atom_vocab, bond_vocab):
    """
    Turn an RDKit molecule into an hgraph of just atoms (no motif or attch config nodes).
    To be used in constructing molecules motif-by-motif during generation.
    """
   
    #Kekulize motifs just like preprocessed mols,
    #or generated mols will have different bonding than them.
    Chem.Kekulize(mol)

    atoms = mol.GetAtoms()
    num_atoms = len(atoms)
    hgraph = get_empty_hierarchical_graph()
    hgraph.add_nodes(num_atoms, ntype = "atom")
    hgraph.nodes["atom"].data["features"] = torch.empty(num_atoms, 2)
    for atom in atoms:
        atom_idx = atom.GetIdx()
        neighbor_idxs = []
        bond_orders = []
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            neighbor_idxs.append(neighbor_idx)
            bond_order = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx).GetBondType()
            bond_orders.append(bond_order)
        update_atom_graph(hgraph, atom, atom_idx,
                          neighbor_idxs, bond_orders)
       
    label_atom_graph(hgraph, atom_vocab, bond_vocab)

    return hgraph

def get_final_graphs(hgraphs_data, atom_vocab, bond_vocab, motif_vocab, attch_config_vocab):
    """
    Label each node and edge of each molecule with the index of its type in the
    corresponding vocabulary. Remove the raw features representing the type from
    the hgraph.
    """

    max_path_length = [0]
    motif_SMILES_to_graph = {}
    fragmented_mol_SMILES = []
    mol_hgraphs = []
    for hgraph_data in tqdm(hgraphs_data):
        get_final_graph(hgraph_data, max_path_length, motif_SMILES_to_graph, fragmented_mol_SMILES, 
                        mol_hgraphs, atom_vocab, bond_vocab, motif_vocab, attch_config_vocab)

    #So ith graph ~ ith motif in vocab.
    motif_graphs = [mol_to_atom_graph(Chem.MolFromSmiles(motif_SMILES), atom_vocab, bond_vocab)
                    for motif_SMILES in motif_vocab]
    max_path_length = max_path_length[0]
    return mol_hgraphs, motif_graphs, max_path_length, fragmented_mol_SMILES

def get_final_vocabs(atom_vocab, bond_vocab, motif_vocab,
                     attch_config_vocab, max_attch_configs,
                     max_num_atoms_in_motif, max_path_length):
    #Include 0 for parent -> child labels.
    ancestry_vocab = torch.arange(max_path_length + 1)
    
    vocabs = {
            "atom": {
                "node": atom_vocab,
                "edge": bond_vocab
            },
            "attachment_config": {
                "node": attch_config_vocab,
                "edge": ancestry_vocab,
                "max_per_motif": max_attch_configs
            },
            "motif": {
                "node": motif_vocab,
                "edge": ancestry_vocab,
                "max_num_atoms": max_num_atoms_in_motif
            }
    }

    return vocabs

def preprocess_mols(mols_SMILES, motif_min_occurences = 0, save_dir = None):
    print("Building vocabs...")
    (hgraphs_data, atom_vocab, bond_vocab, motif_vocab, motif_counter, attch_config_vocab,
     max_attch_configs, max_num_atoms_in_motif) = build_vocabs(mols_SMILES, motif_min_occurences)
    
    print("Getting final graphs...")
    mol_hgraphs, motif_graphs, max_path_length, fragmented_mols_SMILES = get_final_graphs(
        hgraphs_data, atom_vocab, bond_vocab, motif_vocab, attch_config_vocab
    )

    vocabs = get_final_vocabs(atom_vocab, bond_vocab, motif_vocab, attch_config_vocab,
                              max_attch_configs, max_num_atoms_in_motif, max_path_length)
    
    mols_SMILES = { 
        "orig": mols_SMILES,
        "frag": fragmented_mols_SMILES
    }

    if save_dir:
        dgl.data.utils.save_info(str(Path(save_dir)/"mol_SMILES.pkl"), mols_SMILES)
        dgl.data.utils.save_info(str(Path(save_dir)/"vocabs.pkl"), vocabs)
        dgl.data.utils.save_graphs(str(Path(save_dir)/"motif_graphs.bin"), motif_graphs)
        dgl.data.utils.save_graphs(str(Path(save_dir)/"mol_hgraphs.bin"), mol_hgraphs)
    
    return mols_SMILES, vocabs, motif_graphs, mol_hgraphs

def preprocess_mols_data(mols_SMILES, motif_min_occurences = 0, save_dir = None):
    """
    Args:
    mols_SMILES - list of SMILES strings OR
        name of file with SMILES string on each line.
    save_dir - Relative path to the dir to save outputs in.
    """

    if type(mols_SMILES) is str:
        with open(mols_SMILES, "r") as mols_SMILES_file:
            mols_SMILES = [mols_SMILES_line.strip("\n") for mols_SMILES_line in mols_SMILES_file]
            data = preprocess_mols(mols_SMILES, motif_min_occurences, save_dir)
    elif type(mols_SMILES) is list:
        data = preprocess_mols(mols_SMILES, motif_min_occurences, save_dir)
    else:
        raise ValueError("mol_SMILES must be either a list of SMILES strings or a file with one SMILES string on each line!")

    return data
