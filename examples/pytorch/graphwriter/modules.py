import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utlis import *

import dgl.function as fn
from dgl.nn.functional import edge_softmax


class MSA(nn.Module):
    # multi-head self-attention, three modes
    # the first is the copy, determining which entity should be copied.
    # the second is the normal attention with two sequence inputs
    # the third is the attention but with one token and a sequence. (gather, attentive pooling)

    def __init__(self, args, mode="normal"):
        super(MSA, self).__init__()
        if mode == "copy":
            nhead, head_dim = 1, args.nhid
            qninp, kninp = args.dec_ninp, args.nhid
        if mode == "normal":
            nhead, head_dim = args.nhead, args.head_dim
            qninp, kninp = args.nhid, args.nhid
        self.attn_drop = nn.Dropout(0.1)
        self.WQ = nn.Linear(
            qninp, nhead * head_dim, bias=True if mode == "copy" else False
        )
        if mode != "copy":
            self.WK = nn.Linear(kninp, nhead * head_dim, bias=False)
            self.WV = nn.Linear(kninp, nhead * head_dim, bias=False)
        self.args, self.nhead, self.head_dim, self.mode = (
            args,
            nhead,
            head_dim,
            mode,
        )

    def forward(self, inp1, inp2, mask=None):
        B, L2, H = inp2.shape
        NH, HD = self.nhead, self.head_dim
        if self.mode == "copy":
            q, k, v = self.WQ(inp1), inp2, inp2
        else:
            q, k, v = self.WQ(inp1), self.WK(inp2), self.WV(inp2)
        L1 = 1 if inp1.ndim == 2 else inp1.shape[1]
        if self.mode != "copy":
            q = q / math.sqrt(H)
        q = q.view(B, L1, NH, HD).permute(0, 2, 1, 3)
        k = k.view(B, L2, NH, HD).permute(0, 2, 3, 1)
        v = v.view(B, L2, NH, HD).permute(0, 2, 1, 3)
        pre_attn = torch.matmul(q, k)
        if mask is not None:
            pre_attn = pre_attn.masked_fill(mask[:, None, None, :], -1e8)
        if self.mode == "copy":
            return pre_attn.squeeze(1)
        else:
            alpha = self.attn_drop(torch.softmax(pre_attn, -1))
            attn = (
                torch.matmul(alpha, v)
                .permute(0, 2, 1, 3)
                .contiguous()
                .view(B, L1, NH * HD)
            )
            ret = attn
            if inp1.ndim == 2:
                return ret.squeeze(1)
            else:
                return ret


class BiLSTM(nn.Module):
    # for entity encoding or the title encoding
    def __init__(self, args, enc_type="title"):
        super(BiLSTM, self).__init__()
        self.enc_type = enc_type
        self.drop = nn.Dropout(args.emb_drop)
        self.bilstm = nn.LSTM(
            args.nhid,
            args.nhid // 2,
            bidirectional=True,
            num_layers=args.enc_lstm_layers,
            batch_first=True,
        )

    def forward(self, inp, mask, ent_len=None):
        inp = self.drop(inp)
        lens = (mask == 0).sum(-1).long().tolist()
        pad_seq = pack_padded_sequence(
            inp, lens, batch_first=True, enforce_sorted=False
        )
        y, (_h, _c) = self.bilstm(pad_seq)
        if self.enc_type == "title":
            y = pad_packed_sequence(y, batch_first=True)[0]
            return y
        if self.enc_type == "entity":
            _h = _h.transpose(0, 1).contiguous()
            _h = _h[:, -2:].view(
                _h.size(0), -1
            )  # two directions of the top-layer
            ret = pad(_h.split(ent_len), out_type="tensor")
            return ret


class GAT(nn.Module):
    # a graph attention network with dot-product attention
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        ffn_drop=0.0,
        attn_drop=0.0,
        trans=True,
    ):
        super(GAT, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.q_proj = nn.Linear(in_feats, num_heads * out_feats, bias=False)
        self.k_proj = nn.Linear(in_feats, num_heads * out_feats, bias=False)
        self.v_proj = nn.Linear(in_feats, num_heads * out_feats, bias=False)
        self.attn_drop = nn.Dropout(0.1)
        self.ln1 = nn.LayerNorm(in_feats)
        self.ln2 = nn.LayerNorm(in_feats)
        if trans:
            self.FFN = nn.Sequential(
                nn.Linear(in_feats, 4 * in_feats),
                nn.PReLU(4 * in_feats),
                nn.Linear(4 * in_feats, in_feats),
                nn.Dropout(0.1),
            )
            # a strange FFN, see the author's code
        self._trans = trans

    def forward(self, graph, feat):
        graph = graph.local_var()
        feat_c = feat.clone().detach().requires_grad_(False)
        q, k, v = self.q_proj(feat), self.k_proj(feat_c), self.v_proj(feat_c)
        q = q.view(-1, self._num_heads, self._out_feats)
        k = k.view(-1, self._num_heads, self._out_feats)
        v = v.view(-1, self._num_heads, self._out_feats)
        graph.ndata.update(
            {"ft": v, "el": k, "er": q}
        )  # k,q instead of q,k, the edge_softmax is applied on incoming edges
        # compute edge attention
        graph.apply_edges(fn.u_dot_v("el", "er", "e"))
        e = graph.edata.pop("e") / math.sqrt(self._out_feats * self._num_heads)
        graph.edata["a"] = edge_softmax(graph, e)
        # message passing
        graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft2"))
        rst = graph.ndata["ft2"]
        # residual
        rst = rst.view(feat.shape) + feat
        if self._trans:
            rst = self.ln1(rst)
            rst = self.ln1(rst + self.FFN(rst))
            # use the same layer norm, see the author's code
        return rst


class GraphTrans(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.graph_enc == "gat":
            # we only support gtrans, don't use this one
            self.gat = nn.ModuleList(
                [
                    GAT(
                        args.nhid,
                        args.nhid // 4,
                        4,
                        attn_drop=args.attn_drop,
                        trans=False,
                    )
                    for _ in range(args.prop)
                ]
            )  # untested
        else:
            self.gat = nn.ModuleList(
                [
                    GAT(
                        args.nhid,
                        args.nhid // 4,
                        4,
                        attn_drop=args.attn_drop,
                        ffn_drop=args.drop,
                        trans=True,
                    )
                    for _ in range(args.prop)
                ]
            )
        self.prop = args.prop

    def forward(self, ent, ent_mask, ent_len, rel, rel_mask, graphs):
        device = ent.device
        graphs = graphs.to(device)
        ent_mask = ent_mask == 0  # reverse mask
        rel_mask = rel_mask == 0
        init_h = []
        for i in range(graphs.batch_size):
            init_h.append(ent[i][ent_mask[i]])
            init_h.append(rel[i][rel_mask[i]])
        init_h = torch.cat(init_h, 0)
        feats = init_h
        for i in range(self.prop):
            feats = self.gat[i](graphs, feats)
        g_root = feats.index_select(
            0,
            graphs.filter_nodes(
                lambda x: x.data["type"] == NODE_TYPE["root"]
            ).to(device),
        )
        g_ent = pad(
            feats.index_select(
                0,
                graphs.filter_nodes(
                    lambda x: x.data["type"] == NODE_TYPE["entity"]
                ).to(device),
            ).split(ent_len),
            out_type="tensor",
        )
        return g_ent, g_root
