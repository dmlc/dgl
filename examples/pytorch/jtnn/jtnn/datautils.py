import dgl
import torch
from dgl.data.utils import (
    _get_dgl_url,
    download,
    extract_archive,
    get_download_dir,
)
from torch.utils.data import Dataset

from .jtmpn import (
    ATOM_FDIM as ATOM_FDIM_DEC,
    BOND_FDIM as BOND_FDIM_DEC,
    mol2dgl_single as mol2dgl_dec,
)
from .mol_tree import Vocab
from .mol_tree_nx import DGLMolTree
from .mpn import mol2dgl_single as mol2dgl_enc


def _unpack_field(examples, field):
    return [e[field] for e in examples]


def _set_node_id(mol_tree, vocab):
    wid = []
    for i, node in enumerate(mol_tree.nodes_dict):
        mol_tree.nodes_dict[node]["idx"] = i
        wid.append(vocab.get_index(mol_tree.nodes_dict[node]["smiles"]))

    return wid


class JTNNDataset(Dataset):
    def __init__(self, data, vocab, training=True):
        self.dir = get_download_dir()
        self.zip_file_path = "{}/jtnn.zip".format(self.dir)

        download(_get_dgl_url("dgllife/jtnn.zip"), path=self.zip_file_path)
        extract_archive(self.zip_file_path, "{}/jtnn".format(self.dir))
        print("Loading data...")
        data_file = "{}/jtnn/{}.txt".format(self.dir, data)
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]
        self.vocab_file = "{}/jtnn/{}.txt".format(self.dir, vocab)
        print("Loading finished.")
        print("\tNum samples:", len(self.data))
        print("\tVocab file:", self.vocab_file)
        self.training = training
        self.vocab = Vocab([x.strip("\r\n ") for x in open(self.vocab_file)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]
        mol_tree = DGLMolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()

        wid = _set_node_id(mol_tree, self.vocab)

        # prebuild the molecule graph
        mol_graph, atom_x_enc, bond_x_enc = mol2dgl_enc(mol_tree.smiles)

        result = {
            "mol_tree": mol_tree,
            "mol_graph": mol_graph,
            "atom_x_enc": atom_x_enc,
            "bond_x_enc": bond_x_enc,
            "wid": wid,
        }

        if not self.training:
            return result

        # prebuild the candidate graph list
        cands = []
        for node_id, node in mol_tree.nodes_dict.items():
            # fill in ground truth
            if node["label"] not in node["cands"]:
                node["cands"].append(node["label"])
                node["cand_mols"].append(node["label_mol"])

            if node["is_leaf"] or len(node["cands"]) == 1:
                continue
            cands.extend(
                [(cand, mol_tree, node_id) for cand in node["cand_mols"]]
            )
        if len(cands) > 0:
            (
                cand_graphs,
                atom_x_dec,
                bond_x_dec,
                tree_mess_src_e,
                tree_mess_tgt_e,
                tree_mess_tgt_n,
            ) = mol2dgl_dec(cands)
        else:
            cand_graphs = []
            atom_x_dec = torch.zeros(0, ATOM_FDIM_DEC)
            bond_x_dec = torch.zeros(0, BOND_FDIM_DEC)
            tree_mess_src_e = torch.zeros(0, 2).long()
            tree_mess_tgt_e = torch.zeros(0, 2).long()
            tree_mess_tgt_n = torch.zeros(0).long()

        # prebuild the stereoisomers
        cands = mol_tree.stereo_cands
        if len(cands) > 1:
            if mol_tree.smiles3D not in cands:
                cands.append(mol_tree.smiles3D)

            stereo_graphs = [mol2dgl_enc(c) for c in cands]
            stereo_cand_graphs, stereo_atom_x_enc, stereo_bond_x_enc = zip(
                *stereo_graphs
            )
            stereo_atom_x_enc = torch.cat(stereo_atom_x_enc)
            stereo_bond_x_enc = torch.cat(stereo_bond_x_enc)
            stereo_cand_label = [(cands.index(mol_tree.smiles3D), len(cands))]
        else:
            stereo_cand_graphs = []
            stereo_atom_x_enc = torch.zeros(0, atom_x_enc.shape[1])
            stereo_bond_x_enc = torch.zeros(0, bond_x_enc.shape[1])
            stereo_cand_label = []

        result.update(
            {
                "cand_graphs": cand_graphs,
                "atom_x_dec": atom_x_dec,
                "bond_x_dec": bond_x_dec,
                "tree_mess_src_e": tree_mess_src_e,
                "tree_mess_tgt_e": tree_mess_tgt_e,
                "tree_mess_tgt_n": tree_mess_tgt_n,
                "stereo_cand_graphs": stereo_cand_graphs,
                "stereo_atom_x_enc": stereo_atom_x_enc,
                "stereo_bond_x_enc": stereo_bond_x_enc,
                "stereo_cand_label": stereo_cand_label,
            }
        )

        return result


class JTNNCollator(object):
    def __init__(self, vocab, training):
        self.vocab = vocab
        self.training = training

    @staticmethod
    def _batch_and_set(graphs, atom_x, bond_x, flatten):
        if flatten:
            graphs = [g for f in graphs for g in f]
        graph_batch = dgl.batch(graphs)
        graph_batch.ndata["x"] = atom_x
        graph_batch.edata.update(
            {
                "x": bond_x,
                "src_x": atom_x.new(bond_x.shape[0], atom_x.shape[1]).zero_(),
            }
        )
        return graph_batch

    def __call__(self, examples):
        # get list of trees
        mol_trees = _unpack_field(examples, "mol_tree")
        wid = _unpack_field(examples, "wid")
        for _wid, mol_tree in zip(wid, mol_trees):
            mol_tree.graph.ndata["wid"] = torch.LongTensor(_wid)

        # TODO: either support pickling or get around ctypes pointers using scipy
        # batch molecule graphs
        mol_graphs = _unpack_field(examples, "mol_graph")
        atom_x = torch.cat(_unpack_field(examples, "atom_x_enc"))
        bond_x = torch.cat(_unpack_field(examples, "bond_x_enc"))
        mol_graph_batch = self._batch_and_set(mol_graphs, atom_x, bond_x, False)

        result = {
            "mol_trees": mol_trees,
            "mol_graph_batch": mol_graph_batch,
        }

        if not self.training:
            return result

        # batch candidate graphs
        cand_graphs = _unpack_field(examples, "cand_graphs")
        cand_batch_idx = []
        atom_x = torch.cat(_unpack_field(examples, "atom_x_dec"))
        bond_x = torch.cat(_unpack_field(examples, "bond_x_dec"))
        tree_mess_src_e = _unpack_field(examples, "tree_mess_src_e")
        tree_mess_tgt_e = _unpack_field(examples, "tree_mess_tgt_e")
        tree_mess_tgt_n = _unpack_field(examples, "tree_mess_tgt_n")

        n_graph_nodes = 0
        n_tree_nodes = 0
        for i in range(len(cand_graphs)):
            tree_mess_tgt_e[i] += n_graph_nodes
            tree_mess_src_e[i] += n_tree_nodes
            tree_mess_tgt_n[i] += n_graph_nodes
            n_graph_nodes += sum(g.num_nodes() for g in cand_graphs[i])
            n_tree_nodes += mol_trees[i].graph.num_nodes()
            cand_batch_idx.extend([i] * len(cand_graphs[i]))
        tree_mess_tgt_e = torch.cat(tree_mess_tgt_e)
        tree_mess_src_e = torch.cat(tree_mess_src_e)
        tree_mess_tgt_n = torch.cat(tree_mess_tgt_n)

        cand_graph_batch = self._batch_and_set(
            cand_graphs, atom_x, bond_x, True
        )

        # batch stereoisomers
        stereo_cand_graphs = _unpack_field(examples, "stereo_cand_graphs")
        atom_x = torch.cat(_unpack_field(examples, "stereo_atom_x_enc"))
        bond_x = torch.cat(_unpack_field(examples, "stereo_bond_x_enc"))
        stereo_cand_batch_idx = []
        for i in range(len(stereo_cand_graphs)):
            stereo_cand_batch_idx.extend([i] * len(stereo_cand_graphs[i]))

        if len(stereo_cand_batch_idx) > 0:
            stereo_cand_labels = [
                (label, length)
                for ex in _unpack_field(examples, "stereo_cand_label")
                for label, length in ex
            ]
            stereo_cand_labels, stereo_cand_lengths = zip(*stereo_cand_labels)
            stereo_cand_graph_batch = self._batch_and_set(
                stereo_cand_graphs, atom_x, bond_x, True
            )
        else:
            stereo_cand_labels = []
            stereo_cand_lengths = []
            stereo_cand_graph_batch = None
            stereo_cand_batch_idx = []

        result.update(
            {
                "cand_graph_batch": cand_graph_batch,
                "cand_batch_idx": cand_batch_idx,
                "tree_mess_tgt_e": tree_mess_tgt_e,
                "tree_mess_src_e": tree_mess_src_e,
                "tree_mess_tgt_n": tree_mess_tgt_n,
                "stereo_cand_graph_batch": stereo_cand_graph_batch,
                "stereo_cand_batch_idx": stereo_cand_batch_idx,
                "stereo_cand_labels": stereo_cand_labels,
                "stereo_cand_lengths": stereo_cand_lengths,
            }
        )

        return result
