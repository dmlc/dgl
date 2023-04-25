import copy

import rdkit.Chem as Chem
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl import batch, unbatch

from .chemutils import (
    attach_mols_nx,
    copy_edit_mol,
    decode_stereo,
    enum_assemble_nx,
    set_atommap,
)
from .jtmpn import DGLJTMPN, mol2dgl_single as mol2dgl_dec
from .jtnn_dec import DGLJTNNDecoder
from .jtnn_enc import DGLJTNNEncoder
from .mpn import DGLMPN, mol2dgl_single as mol2dgl_enc
from .nnutils import cuda


class DGLJTNNVAE(nn.Module):
    def __init__(self, vocab, hidden_size, latent_size, depth):
        super(DGLJTNNVAE, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth

        self.embedding = nn.Embedding(vocab.size(), hidden_size)
        self.mpn = DGLMPN(hidden_size, depth)
        self.jtnn = DGLJTNNEncoder(vocab, hidden_size, self.embedding)
        self.decoder = DGLJTNNDecoder(
            vocab, hidden_size, latent_size // 2, self.embedding
        )
        self.jtmpn = DGLJTMPN(hidden_size, depth)

        self.T_mean = nn.Linear(hidden_size, latent_size // 2)
        self.T_var = nn.Linear(hidden_size, latent_size // 2)
        self.G_mean = nn.Linear(hidden_size, latent_size // 2)
        self.G_var = nn.Linear(hidden_size, latent_size // 2)

        self.n_nodes_total = 0
        self.n_passes = 0
        self.n_edges_total = 0
        self.n_tree_nodes_total = 0

    @staticmethod
    def move_to_cuda(mol_batch):
        for i in range(len(mol_batch["mol_trees"])):
            mol_batch["mol_trees"][i].graph = cuda(
                mol_batch["mol_trees"][i].graph
            )

        mol_batch["mol_graph_batch"] = cuda(mol_batch["mol_graph_batch"])
        if "cand_graph_batch" in mol_batch:
            mol_batch["cand_graph_batch"] = cuda(mol_batch["cand_graph_batch"])
        if mol_batch.get("stereo_cand_graph_batch") is not None:
            mol_batch["stereo_cand_graph_batch"] = cuda(
                mol_batch["stereo_cand_graph_batch"]
            )

    def encode(self, mol_batch):
        mol_graphs = mol_batch["mol_graph_batch"]
        mol_vec = self.mpn(mol_graphs)

        mol_tree_batch, tree_vec = self.jtnn(
            [t.graph for t in mol_batch["mol_trees"]]
        )

        self.n_nodes_total += mol_graphs.num_nodes()
        self.n_edges_total += mol_graphs.num_edges()
        self.n_tree_nodes_total += sum(
            t.graph.num_nodes() for t in mol_batch["mol_trees"]
        )
        self.n_passes += 1

        return mol_tree_batch, tree_vec, mol_vec

    def sample(self, tree_vec, mol_vec, e1=None, e2=None):
        tree_mean = self.T_mean(tree_vec)
        tree_log_var = -torch.abs(self.T_var(tree_vec))
        mol_mean = self.G_mean(mol_vec)
        mol_log_var = -torch.abs(self.G_var(mol_vec))

        epsilon = cuda(torch.randn(*tree_mean.shape)) if e1 is None else e1
        tree_vec = tree_mean + torch.exp(tree_log_var / 2) * epsilon
        epsilon = cuda(torch.randn(*mol_mean.shape)) if e2 is None else e2
        mol_vec = mol_mean + torch.exp(mol_log_var / 2) * epsilon

        z_mean = torch.cat([tree_mean, mol_mean], 1)
        z_log_var = torch.cat([tree_log_var, mol_log_var], 1)

        return tree_vec, mol_vec, z_mean, z_log_var

    def forward(self, mol_batch, beta=0, e1=None, e2=None):
        self.move_to_cuda(mol_batch)

        mol_trees = mol_batch["mol_trees"]
        batch_size = len(mol_trees)

        mol_tree_batch, tree_vec, mol_vec = self.encode(mol_batch)

        tree_vec, mol_vec, z_mean, z_log_var = self.sample(
            tree_vec, mol_vec, e1, e2
        )
        kl_loss = (
            -0.5
            * torch.sum(
                1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)
            )
            / batch_size
        )

        word_loss, topo_loss, word_acc, topo_acc = self.decoder(
            [t.graph for t in mol_trees], tree_vec
        )
        assm_loss, assm_acc = self.assm(mol_batch, mol_tree_batch, mol_vec)
        stereo_loss, stereo_acc = self.stereo(mol_batch, mol_vec)

        loss = (
            word_loss + topo_loss + assm_loss + 2 * stereo_loss + beta * kl_loss
        )

        return loss, kl_loss, word_acc, topo_acc, assm_acc, stereo_acc

    def assm(self, mol_batch, mol_tree_batch, mol_vec):
        cands = [
            mol_batch["cand_graph_batch"],
            cuda(mol_batch["tree_mess_src_e"]),
            cuda(mol_batch["tree_mess_tgt_e"]),
            cuda(mol_batch["tree_mess_tgt_n"]),
        ]
        cand_vec = self.jtmpn(cands, mol_tree_batch)
        cand_vec = self.G_mean(cand_vec)

        batch_idx = cuda(torch.LongTensor(mol_batch["cand_batch_idx"]))
        mol_vec = mol_vec[batch_idx]

        mol_vec = mol_vec.view(-1, 1, self.latent_size // 2)
        cand_vec = cand_vec.view(-1, self.latent_size // 2, 1)
        scores = (mol_vec @ cand_vec)[:, 0, 0]

        cnt, tot, acc = 0, 0, 0
        all_loss = []
        for i, mol_tree in enumerate(mol_batch["mol_trees"]):
            comp_nodes = [
                node_id
                for node_id, node in mol_tree.nodes_dict.items()
                if len(node["cands"]) > 1 and not node["is_leaf"]
            ]
            cnt += len(comp_nodes)
            # segmented accuracy and cross entropy
            for node_id in comp_nodes:
                node = mol_tree.nodes_dict[node_id]
                label = node["cands"].index(node["label"])
                ncand = len(node["cands"])
                cur_score = scores[tot : tot + ncand]
                tot += ncand

                if cur_score[label].item() >= cur_score.max().item():
                    acc += 1

                label = cuda(torch.LongTensor([label]))
                all_loss.append(
                    F.cross_entropy(
                        cur_score.view(1, -1), label, size_average=False
                    )
                )

        all_loss = sum(all_loss) / len(mol_batch["mol_trees"])
        return all_loss, acc / cnt

    def stereo(self, mol_batch, mol_vec):
        stereo_cands = mol_batch["stereo_cand_graph_batch"]
        batch_idx = mol_batch["stereo_cand_batch_idx"]
        labels = mol_batch["stereo_cand_labels"]
        lengths = mol_batch["stereo_cand_lengths"]

        if len(labels) == 0:
            # Only one stereoisomer exists; do nothing
            return cuda(torch.tensor(0.0)), 1.0

        batch_idx = cuda(torch.LongTensor(batch_idx))
        stereo_cands = self.mpn(stereo_cands)
        stereo_cands = self.G_mean(stereo_cands)
        stereo_labels = mol_vec[batch_idx]
        scores = F.cosine_similarity(stereo_cands, stereo_labels)

        st, acc = 0, 0
        all_loss = []
        for label, le in zip(labels, lengths):
            cur_scores = scores[st : st + le]
            if cur_scores.data[label].item() >= cur_scores.max().item():
                acc += 1
            label = cuda(torch.LongTensor([label]))
            all_loss.append(
                F.cross_entropy(
                    cur_scores.view(1, -1), label, size_average=False
                )
            )
            st += le

        all_loss = sum(all_loss) / len(labels)
        return all_loss, acc / len(labels)

    def decode(self, tree_vec, mol_vec):
        mol_tree, nodes_dict, effective_nodes = self.decoder.decode(tree_vec)
        effective_nodes_list = effective_nodes.tolist()
        nodes_dict = [nodes_dict[v] for v in effective_nodes_list]

        for i, (node_id, node) in enumerate(
            zip(effective_nodes_list, nodes_dict)
        ):
            node["idx"] = i
            node["nid"] = i + 1
            node["is_leaf"] = True
            if mol_tree.graph.in_degrees(node_id) > 1:
                node["is_leaf"] = False
                set_atommap(node["mol"], node["nid"])

        mol_tree_sg = mol_tree.graph.subgraph(
            effective_nodes.to(tree_vec.device)
        )
        mol_tree_msg, _ = self.jtnn([mol_tree_sg])
        mol_tree_msg = unbatch(mol_tree_msg)[0]
        mol_tree_msg.nodes_dict = nodes_dict

        cur_mol = copy_edit_mol(nodes_dict[0]["mol"])
        global_amap = [{}] + [{} for node in nodes_dict]
        global_amap[1] = {
            atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()
        }

        cur_mol = self.dfs_assemble(
            mol_tree_msg, mol_vec, cur_mol, global_amap, [], 0, None
        )
        if cur_mol is None:
            return None

        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        if cur_mol is None:
            return None

        smiles2D = Chem.MolToSmiles(cur_mol)
        stereo_cands = decode_stereo(smiles2D)
        if len(stereo_cands) == 1:
            return stereo_cands[0]
        stereo_graphs = [mol2dgl_enc(c) for c in stereo_cands]
        stereo_cand_graphs, atom_x, bond_x = zip(*stereo_graphs)
        stereo_cand_graphs = cuda(batch(stereo_cand_graphs))
        atom_x = cuda(torch.cat(atom_x))
        bond_x = cuda(torch.cat(bond_x))
        stereo_cand_graphs.ndata["x"] = atom_x
        stereo_cand_graphs.edata["x"] = bond_x
        stereo_cand_graphs.edata["src_x"] = atom_x.new(
            bond_x.shape[0], atom_x.shape[1]
        ).zero_()
        stereo_vecs = self.mpn(stereo_cand_graphs)
        stereo_vecs = self.G_mean(stereo_vecs)
        scores = F.cosine_similarity(stereo_vecs, mol_vec)
        _, max_id = scores.max(0)
        return stereo_cands[max_id.item()]

    def dfs_assemble(
        self,
        mol_tree_msg,
        mol_vec,
        cur_mol,
        global_amap,
        fa_amap,
        cur_node_id,
        fa_node_id,
    ):
        nodes_dict = mol_tree_msg.nodes_dict
        fa_node = nodes_dict[fa_node_id] if fa_node_id is not None else None
        cur_node = nodes_dict[cur_node_id]

        fa_nid = fa_node["nid"] if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children_node_id = [
            v
            for v in mol_tree_msg.successors(cur_node_id).tolist()
            if nodes_dict[v]["nid"] != fa_nid
        ]
        children = [nodes_dict[v] for v in children_node_id]
        neighbors = [nei for nei in children if nei["mol"].GetNumAtoms() > 1]
        neighbors = sorted(
            neighbors, key=lambda x: x["mol"].GetNumAtoms(), reverse=True
        )
        singletons = [nei for nei in children if nei["mol"].GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [
            (fa_nid, a2, a1)
            for nid, a1, a2 in fa_amap
            if nid == cur_node["nid"]
        ]
        cands = enum_assemble_nx(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0:
            return None
        cand_smiles, cand_mols, cand_amap = list(zip(*cands))

        cands = [(candmol, mol_tree_msg, cur_node_id) for candmol in cand_mols]
        (
            cand_graphs,
            atom_x,
            bond_x,
            tree_mess_src_edges,
            tree_mess_tgt_edges,
            tree_mess_tgt_nodes,
        ) = mol2dgl_dec(cands)
        cand_graphs = batch([g.to(mol_vec.device) for g in cand_graphs])
        atom_x = cuda(atom_x)
        bond_x = cuda(bond_x)
        cand_graphs.ndata["x"] = atom_x
        cand_graphs.edata["x"] = bond_x
        cand_graphs.edata["src_x"] = atom_x.new(
            bond_x.shape[0], atom_x.shape[1]
        ).zero_()

        cand_vecs = self.jtmpn(
            (
                cand_graphs,
                tree_mess_src_edges,
                tree_mess_tgt_edges,
                tree_mess_tgt_nodes,
            ),
            mol_tree_msg,
        )
        cand_vecs = self.G_mean(cand_vecs)
        mol_vec = mol_vec.squeeze()
        scores = cand_vecs @ mol_vec

        _, cand_idx = torch.sort(scores, descending=True)

        backup_mol = Chem.RWMol(cur_mol)
        for i in range(len(cand_idx)):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item()]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id, ctr_atom, nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[
                    cur_node["nid"]
                ][ctr_atom]

            cur_mol = attach_mols_nx(cur_mol, children, [], new_global_amap)
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None:
                continue

            result = True
            for nei_node_id, nei_node in zip(children_node_id, children):
                if nei_node["is_leaf"]:
                    continue
                cur_mol = self.dfs_assemble(
                    mol_tree_msg,
                    mol_vec,
                    cur_mol,
                    new_global_amap,
                    pred_amap,
                    nei_node_id,
                    cur_node_id,
                )
                if cur_mol is None:
                    result = False
                    break

            if result:
                return cur_mol

        return None
