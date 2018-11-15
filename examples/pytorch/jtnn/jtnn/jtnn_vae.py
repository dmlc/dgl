import torch
import torch.nn as nn
import torch.nn.functional as F
from .mol_tree import Vocab
from .nnutils import create_var, cuda
from .jtnn_enc import DGLJTNNEncoder
from .jtnn_dec import DGLJTNNDecoder
from .mpn import DGLMPN, mol2dgl
from .jtmpn import DGLJTMPN
from .line_profiler_integration import profile

import rdkit
import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import copy, math

def dgl_set_batch_nodeID(mol_batch, vocab):
    for mol_tree in mol_batch:
        wid = []
        for i, node in enumerate(mol_tree.nodes_dict):
            mol_tree.nodes_dict[node]['idx'] = i
            wid.append(vocab.get_index(mol_tree.nodes_dict[node]['smiles']))
        mol_tree.ndata['wid'] = cuda(torch.LongTensor(wid))

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
                vocab, hidden_size, latent_size // 2, self.embedding)
        self.jtmpn = DGLJTMPN(hidden_size, depth)

        self.T_mean = nn.Linear(hidden_size, latent_size // 2)
        self.T_var = nn.Linear(hidden_size, latent_size // 2)
        self.G_mean = nn.Linear(hidden_size, latent_size // 2)
        self.G_var = nn.Linear(hidden_size, latent_size // 2)

        self.n_nodes_total = 0
        self.n_passes = 0
        self.n_edges_total = 0
        self.n_tree_nodes_total = 0

    def encode(self, mol_batch):
        mol_graphs = mol_batch['mol_graph_batch']
        mol_vec = self.mpn(mol_graphs)

        mol_tree_batch, tree_vec = self.jtnn(mol_batch['mol_trees'])

        self.n_nodes_total += mol_graphs.number_of_nodes()
        self.n_edges_total += mol_graphs.number_of_edges()
        self.n_tree_nodes_total += sum(t.number_of_nodes() for t in mol_batch['mol_trees'])
        self.n_passes += 1

        return mol_tree_batch, tree_vec, mol_vec

    def forward(self, mol_batch, beta=0, e1=None, e2=None):
        mol_trees = mol_batch['mol_trees']
        batch_size = len(mol_trees)

        mol_tree_batch, tree_vec, mol_vec = self.encode(mol_batch)

        tree_mean = self.T_mean(tree_vec)
        tree_log_var = -torch.abs(self.T_var(tree_vec))
        mol_mean = self.G_mean(mol_vec)
        mol_log_var = -torch.abs(self.G_var(mol_vec))

        z_mean = torch.cat([tree_mean, mol_mean], dim=1)
        z_log_var = torch.cat([tree_log_var, mol_log_var], dim=1)
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size

        self.z_mean = z_mean
        self.z_log_var = z_log_var

        epsilon = cuda(torch.randn(batch_size, self.latent_size // 2)) if e1 is None else e1
        tree_vec = tree_mean + torch.exp(tree_log_var / 2) * epsilon
        epsilon = cuda(torch.randn(batch_size, self.latent_size // 2)) if e2 is None else e2
        mol_vec = mol_mean + torch.exp(mol_log_var / 2) * epsilon
        self.tree_mean = tree_mean
        self.tree_log_var = tree_log_var
        self.e1 = epsilon
        self.tree_vec = tree_vec

        word_loss, topo_loss, word_acc, topo_acc = self.decoder(mol_trees, tree_vec)
        assm_loss, assm_acc = self.assm(mol_batch, mol_tree_batch, mol_vec)
        stereo_loss, stereo_acc = self.stereo(mol_batch, mol_vec)

        self.word_loss_v = word_loss
        self.topo_loss_v = topo_loss
        self.assm_loss_v = assm_loss
        self.stereo_loss_v = stereo_loss

        all_vec = torch.cat([tree_vec, mol_vec], dim=1)
        loss = word_loss + topo_loss + assm_loss + 2 * stereo_loss + beta * kl_loss

        self.all_vec = all_vec

        return loss, kl_loss, word_acc, topo_acc, assm_acc, stereo_acc

    def assm(self, mol_batch, mol_tree_batch, mol_vec):
        cands = [mol_batch['cand_graph_batch'],
                 mol_batch['tree_mess_src_e'],
                 mol_batch['tree_mess_tgt_e'],
                 mol_batch['tree_mess_tgt_n']]
        cand_vec = self.jtmpn(cands, mol_tree_batch)
        cand_vec = self.G_mean(cand_vec)

        batch_idx = cuda(torch.LongTensor(mol_batch['cand_batch_idx']))
        mol_vec = mol_vec[batch_idx]
        self.mol_vec_v = mol_vec

        mol_vec = mol_vec.view(-1, 1, self.latent_size // 2)
        cand_vec = cand_vec.view(-1, self.latent_size // 2, 1)
        scores = (mol_vec @ cand_vec)[:, 0, 0]
        self.scores = scores

        cnt, tot, acc = 0, 0, 0
        all_loss = []
        for i, mol_tree in enumerate(mol_batch['mol_trees']):
            comp_nodes = [node_id for node_id, node in mol_tree.nodes_dict.items()
                          if len(node['cands']) > 1 and not node['is_leaf']]
            cnt += len(comp_nodes)
            # segmented accuracy and cross entropy
            for node_id in comp_nodes:
                node = mol_tree.nodes_dict[node_id]
                label = node['cands'].index(node['label'])
                ncand = len(node['cands'])
                cur_score = scores[tot:tot+ncand]
                tot += ncand

                if cur_score[label].item() >= cur_score.max().item():
                    acc += 1

                label = cuda(torch.LongTensor([label]))
                all_loss.append(
                    F.cross_entropy(cur_score.view(1, -1), label, size_average=False))

        all_loss = sum(all_loss) / len(mol_batch['mol_trees'])
        return all_loss, acc / cnt

    def stereo(self, mol_batch, mol_vec):
        stereo_cands = mol_batch['stereo_cand_graph_batch']
        batch_idx = mol_batch['stereo_cand_batch_idx']
        labels = mol_batch['stereo_cand_labels']
        lengths = mol_batch['stereo_cand_lengths']

        if len(labels) == 0:
            # Only one stereoisomer exists; do nothing
            return cuda(torch.tensor(0.)), 1.

        batch_idx = cuda(torch.LongTensor(batch_idx))
        stereo_cands = self.mpn(stereo_cands)
        stereo_cands = self.G_mean(stereo_cands)
        stereo_labels = mol_vec[batch_idx]
        scores = F.cosine_similarity(stereo_cands, stereo_labels)

        st, acc = 0, 0
        all_loss = []
        for label, le in zip(labels, lengths):
            cur_scores = scores[st:st+le]
            if cur_scores.data[label].item() >= cur_scores.max().item():
                acc += 1
            label = cuda(torch.LongTensor([label]))
            all_loss.append(
                F.cross_entropy(cur_scores.view(1, -1), label, size_average=False))
            st += le

        all_loss = sum(all_loss) / len(labels)
        return all_loss, acc / len(labels)
