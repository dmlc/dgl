import torch
from modules import BiLSTM, GraphTrans, MSA
from torch import nn
from utlis import *

import dgl


class GraphWriter(nn.Module):
    def __init__(self, args):
        super(GraphWriter, self).__init__()
        self.args = args
        if args.title:
            self.title_emb = nn.Embedding(
                len(args.title_vocab), args.nhid, padding_idx=0
            )
            self.title_enc = BiLSTM(args, enc_type="title")
            self.title_attn = MSA(args)
        self.ent_emb = nn.Embedding(
            len(args.ent_text_vocab), args.nhid, padding_idx=0
        )
        self.tar_emb = nn.Embedding(
            len(args.text_vocab), args.nhid, padding_idx=0
        )
        if args.title:
            nn.init.xavier_normal_(self.title_emb.weight)
        nn.init.xavier_normal_(self.ent_emb.weight)
        self.rel_emb = nn.Embedding(
            len(args.rel_vocab), args.nhid, padding_idx=0
        )
        nn.init.xavier_normal_(self.rel_emb.weight)
        self.decode_lstm = nn.LSTMCell(args.dec_ninp, args.nhid)
        self.ent_enc = BiLSTM(args, enc_type="entity")
        self.graph_enc = GraphTrans(args)
        self.ent_attn = MSA(args)
        self.copy_attn = MSA(args, mode="copy")
        self.copy_fc = nn.Linear(args.dec_ninp, 1)
        self.pred_v_fc = nn.Linear(args.dec_ninp, len(args.text_vocab))

    def enc_forward(
        self, batch, ent_mask, ent_text_mask, ent_len, rel_mask, title_mask
    ):
        title_enc = None
        if self.args.title:
            title_enc = self.title_enc(
                self.title_emb(batch["title"]), title_mask
            )
        ent_enc = self.ent_enc(
            self.ent_emb(batch["ent_text"]),
            ent_text_mask,
            ent_len=batch["ent_len"],
        )
        rel_emb = self.rel_emb(batch["rel"])
        g_ent, g_root = self.graph_enc(
            ent_enc, ent_mask, ent_len, rel_emb, rel_mask, batch["graph"]
        )
        return g_ent, g_root, title_enc, ent_enc

    def forward(self, batch, beam_size=-1):
        ent_mask = len2mask(batch["ent_len"], self.args.device)
        ent_text_mask = batch["ent_text"] == 0
        rel_mask = batch["rel"] == 0  # 0 means the <PAD>
        title_mask = batch["title"] == 0
        g_ent, g_root, title_enc, ent_enc = self.enc_forward(
            batch,
            ent_mask,
            ent_text_mask,
            batch["ent_len"],
            rel_mask,
            title_mask,
        )

        _h, _c = g_root, g_root.clone().detach()
        ctx = _h + self.ent_attn(_h, g_ent, mask=ent_mask)
        if self.args.title:
            attn = _h + self.title_attn(_h, title_enc, mask=title_mask)
            ctx = torch.cat([ctx, attn], 1)
        if beam_size < 1:
            # training
            outs = []
            tar_inp = self.tar_emb(batch["text"].transpose(0, 1))
            for t, xt in enumerate(tar_inp):
                _xt = torch.cat([ctx, xt], 1)
                _h, _c = self.decode_lstm(_xt, (_h, _c))
                ctx = _h + self.ent_attn(_h, g_ent, mask=ent_mask)
                if self.args.title:
                    attn = _h + self.title_attn(_h, title_enc, mask=title_mask)
                    ctx = torch.cat([ctx, attn], 1)
                outs.append(torch.cat([_h, ctx], 1))
            outs = torch.stack(outs, 1)
            copy_gate = torch.sigmoid(self.copy_fc(outs))
            EPSI = 1e-6
            # copy
            pred_v = torch.log(copy_gate + EPSI) + torch.log_softmax(
                self.pred_v_fc(outs), -1
            )
            pred_c = torch.log((1.0 - copy_gate) + EPSI) + torch.log_softmax(
                self.copy_attn(outs, ent_enc, mask=ent_mask), -1
            )
            pred = torch.cat([pred_v, pred_c], -1)
            return pred
        else:
            if beam_size == 1:
                # greedy
                device = g_ent.device
                B = g_ent.shape[0]
                ent_type = batch["ent_type"].view(B, -1)
                seq = (
                    torch.ones(
                        B,
                    )
                    .long()
                    .to(device)
                    * self.args.text_vocab("<BOS>")
                ).unsqueeze(1)
                for t in range(self.args.beam_max_len):
                    _inp = replace_ent(
                        seq[:, -1], ent_type, len(self.args.text_vocab)
                    )
                    xt = self.tar_emb(_inp)
                    _xt = torch.cat([ctx, xt], 1)
                    _h, _c = self.decode_lstm(_xt, (_h, _c))
                    ctx = _h + self.ent_attn(_h, g_ent, mask=ent_mask)
                    if self.args.title:
                        attn = _h + self.title_attn(
                            _h, title_enc, mask=title_mask
                        )
                        ctx = torch.cat([ctx, attn], 1)
                    _y = torch.cat([_h, ctx], 1)
                    copy_gate = torch.sigmoid(self.copy_fc(_y))
                    pred_v = torch.log(copy_gate) + torch.log_softmax(
                        self.pred_v_fc(_y), -1
                    )
                    pred_c = torch.log((1.0 - copy_gate)) + torch.log_softmax(
                        self.copy_attn(
                            _y.unsqueeze(1), ent_enc, mask=ent_mask
                        ).squeeze(1),
                        -1,
                    )
                    pred = torch.cat([pred_v, pred_c], -1).view(B, -1)
                    for ban_item in ["<BOS>", "<PAD>", "<UNK>"]:
                        pred[:, self.args.text_vocab(ban_item)] = -1e8
                    _, word = pred.max(-1)
                    seq = torch.cat([seq, word.unsqueeze(1)], 1)
                return seq
            else:
                # beam search
                device = g_ent.device
                B = g_ent.shape[0]
                BSZ = B * beam_size
                _h = _h.view(B, 1, -1).repeat(1, beam_size, 1).view(BSZ, -1)
                _c = _c.view(B, 1, -1).repeat(1, beam_size, 1).view(BSZ, -1)
                ent_mask = (
                    ent_mask.view(B, 1, -1)
                    .repeat(1, beam_size, 1)
                    .view(BSZ, -1)
                )
                if self.args.title:
                    title_mask = (
                        title_mask.view(B, 1, -1)
                        .repeat(1, beam_size, 1)
                        .view(BSZ, -1)
                    )
                    title_enc = (
                        title_enc.view(B, 1, title_enc.size(1), -1)
                        .repeat(1, beam_size, 1, 1)
                        .view(BSZ, title_enc.size(1), -1)
                    )
                ctx = ctx.view(B, 1, -1).repeat(1, beam_size, 1).view(BSZ, -1)
                ent_type = (
                    batch["ent_type"]
                    .view(B, 1, -1)
                    .repeat(1, beam_size, 1)
                    .view(BSZ, -1)
                )
                g_ent = (
                    g_ent.view(B, 1, g_ent.size(1), -1)
                    .repeat(1, beam_size, 1, 1)
                    .view(BSZ, g_ent.size(1), -1)
                )
                ent_enc = (
                    ent_enc.view(B, 1, ent_enc.size(1), -1)
                    .repeat(1, beam_size, 1, 1)
                    .view(BSZ, ent_enc.size(1), -1)
                )

                beam_best = torch.zeros(B).to(device) - 1e9
                beam_best_seq = [None] * B
                beam_seq = (
                    torch.ones(B, beam_size).long().to(device)
                    * self.args.text_vocab("<BOS>")
                ).unsqueeze(-1)
                beam_score = torch.zeros(B, beam_size).to(device)
                done_flag = torch.zeros(B, beam_size)
                for t in range(self.args.beam_max_len):
                    _inp = replace_ent(
                        beam_seq[:, :, -1].view(-1),
                        ent_type,
                        len(self.args.text_vocab),
                    )
                    xt = self.tar_emb(_inp)
                    _xt = torch.cat([ctx, xt], 1)
                    _h, _c = self.decode_lstm(_xt, (_h, _c))
                    ctx = _h + self.ent_attn(_h, g_ent, mask=ent_mask)
                    if self.args.title:
                        attn = _h + self.title_attn(
                            _h, title_enc, mask=title_mask
                        )
                        ctx = torch.cat([ctx, attn], 1)
                    _y = torch.cat([_h, ctx], 1)
                    copy_gate = torch.sigmoid(self.copy_fc(_y))
                    pred_v = torch.log(copy_gate) + torch.log_softmax(
                        self.pred_v_fc(_y), -1
                    )
                    pred_c = torch.log((1.0 - copy_gate)) + torch.log_softmax(
                        self.copy_attn(
                            _y.unsqueeze(1), ent_enc, mask=ent_mask
                        ).squeeze(1),
                        -1,
                    )
                    pred = torch.cat([pred_v, pred_c], -1).view(
                        B, beam_size, -1
                    )
                    for ban_item in ["<BOS>", "<PAD>", "<UNK>"]:
                        pred[:, :, self.args.text_vocab(ban_item)] = -1e8
                    if t == self.args.beam_max_len - 1:  # force ending
                        tt = pred[:, :, self.args.text_vocab("<EOS>")]
                        pred = pred * 0 - 1e8
                        pred[:, :, self.args.text_vocab("<EOS>")] = tt
                    cum_score = beam_score.view(B, beam_size, 1) + pred
                    score, word = cum_score.topk(
                        dim=-1, k=beam_size
                    )  # B, beam_size, beam_size
                    score, word = score.view(B, -1), word.view(B, -1)
                    eos_idx = self.args.text_vocab("<EOS>")
                    if beam_seq.size(2) == 1:
                        new_idx = torch.arange(beam_size).to(word)
                        new_idx = new_idx[None, :].repeat(B, 1)
                    else:
                        _, new_idx = score.topk(dim=-1, k=beam_size)
                    new_src, new_score, new_word, new_done = [], [], [], []
                    LP = beam_seq.size(2) ** self.args.lp
                    for i in range(B):
                        for j in range(beam_size):
                            tmp_score = score[i][new_idx[i][j]]
                            tmp_word = word[i][new_idx[i][j]]
                            src_idx = new_idx[i][j] // beam_size
                            new_src.append(src_idx)
                            if tmp_word == eos_idx:
                                new_score.append(-1e8)
                            else:
                                new_score.append(tmp_score)
                            new_word.append(tmp_word)

                            if (
                                tmp_word == eos_idx
                                and done_flag[i][src_idx] == 0
                                and tmp_score / LP > beam_best[i]
                            ):
                                beam_best[i] = tmp_score / LP
                                beam_best_seq[i] = beam_seq[i][src_idx]
                            if tmp_word == eos_idx:
                                new_done.append(1)
                            else:
                                new_done.append(done_flag[i][src_idx])
                    new_score = (
                        torch.Tensor(new_score)
                        .view(B, beam_size)
                        .to(beam_score)
                    )
                    new_word = (
                        torch.Tensor(new_word).view(B, beam_size).to(beam_seq)
                    )
                    new_src = (
                        torch.LongTensor(new_src).view(B, beam_size).to(device)
                    )
                    new_done = (
                        torch.Tensor(new_done).view(B, beam_size).to(done_flag)
                    )
                    beam_score = new_score
                    done_flag = new_done
                    beam_seq = beam_seq.view(B, beam_size, -1)[
                        torch.arange(B)[:, None].to(device), new_src
                    ]
                    beam_seq = torch.cat([beam_seq, new_word.unsqueeze(2)], 2)
                    _h = _h.view(B, beam_size, -1)[
                        torch.arange(B)[:, None].to(device), new_src
                    ].view(BSZ, -1)
                    _c = _c.view(B, beam_size, -1)[
                        torch.arange(B)[:, None].to(device), new_src
                    ].view(BSZ, -1)
                    ctx = ctx.view(B, beam_size, -1)[
                        torch.arange(B)[:, None].to(device), new_src
                    ].view(BSZ, -1)

                return beam_best_seq
