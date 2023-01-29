import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphwriter import *
from opts import *
from tqdm import tqdm
from utlis import *

sys.path.append("./pycocoevalcap")
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


def train_one_epoch(model, dataloader, optimizer, args, epoch):
    model.train()
    tloss = 0.0
    tcnt = 0.0
    st_time = time.time()
    with tqdm(dataloader, desc="Train Ep " + str(epoch), mininterval=60) as tq:
        for batch in tq:
            pred = model(batch)
            nll_loss = F.nll_loss(
                pred.view(-1, pred.shape[-1]),
                batch["tgt_text"].view(-1),
                ignore_index=0,
            )
            loss = nll_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            loss = loss.item()
            if loss != loss:
                raise ValueError("NaN appear")
            tloss += loss * len(batch["tgt_text"])
            tcnt += len(batch["tgt_text"])
            tq.set_postfix({"loss": tloss / tcnt}, refresh=False)
    print(
        "Train Ep ",
        str(epoch),
        "AVG Loss ",
        tloss / tcnt,
        "Steps ",
        tcnt,
        "Time ",
        time.time() - st_time,
        "GPU",
        torch.cuda.max_memory_cached() / 1024.0 / 1024.0 / 1024.0,
    )
    torch.save(model, args.save_model + str(epoch % 100))


val_loss = 2**31


def eval_it(model, dataloader, args, epoch):
    global val_loss
    model.eval()
    tloss = 0.0
    tcnt = 0.0
    st_time = time.time()
    with tqdm(dataloader, desc="Eval Ep " + str(epoch), mininterval=60) as tq:
        for batch in tq:
            with torch.no_grad():
                pred = model(batch)
                nll_loss = F.nll_loss(
                    pred.view(-1, pred.shape[-1]),
                    batch["tgt_text"].view(-1),
                    ignore_index=0,
                )
            loss = nll_loss
            loss = loss.item()
            tloss += loss * len(batch["tgt_text"])
            tcnt += len(batch["tgt_text"])
            tq.set_postfix({"loss": tloss / tcnt}, refresh=False)
    print(
        "Eval Ep ",
        str(epoch),
        "AVG Loss ",
        tloss / tcnt,
        "Steps ",
        tcnt,
        "Time ",
        time.time() - st_time,
    )
    if tloss / tcnt < val_loss:
        print("Saving best model ", "Ep ", epoch, " loss ", tloss / tcnt)
        torch.save(model, args.save_model + "best")
        val_loss = tloss / tcnt


def test(model, dataloader, args):
    scorer = Bleu(4)
    m_scorer = Meteor()
    r_scorer = Rouge()
    hyp = []
    ref = []
    model.eval()
    gold_file = open("tmp_gold.txt", "w")
    pred_file = open("tmp_pred.txt", "w")
    with tqdm(dataloader, desc="Test ", mininterval=1) as tq:
        for batch in tq:
            with torch.no_grad():
                seq = model(batch, beam_size=args.beam_size)
            r = write_txt(batch, batch["tgt_text"], gold_file, args)
            h = write_txt(batch, seq, pred_file, args)
            hyp.extend(h)
            ref.extend(r)
    hyp = dict(zip(range(len(hyp)), hyp))
    ref = dict(zip(range(len(ref)), ref))
    print(hyp[0], ref[0])
    print("BLEU INP", len(hyp), len(ref))
    print("BLEU", scorer.compute_score(ref, hyp)[0])
    print("METEOR", m_scorer.compute_score(ref, hyp)[0])
    print("ROUGE_L", r_scorer.compute_score(ref, hyp)[0])
    gold_file.close()
    pred_file.close()


def main(args):
    if os.path.exists(args.save_dataset):
        train_dataset, valid_dataset, test_dataset = pickle.load(
            open(args.save_dataset, "rb")
        )
    else:
        train_dataset, valid_dataset, test_dataset = get_datasets(
            args.fnames, device=args.device, save=args.save_dataset
        )
    args = vocab_config(
        args,
        train_dataset.ent_vocab,
        train_dataset.rel_vocab,
        train_dataset.text_vocab,
        train_dataset.ent_text_vocab,
        train_dataset.title_vocab,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=BucketSampler(train_dataset, batch_size=args.batch_size),
        collate_fn=train_dataset.batch_fn,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_dataset.batch_fn,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_dataset.batch_fn,
    )

    model = GraphWriter(args)
    model.to(args.device)
    if args.test:
        model = torch.load(args.save_model)
        model.args = args
        print(model)
        test(model, test_dataloader, args)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=0.9,
        )
        print(model)
        for epoch in range(args.epoch):
            train_one_epoch(model, train_dataloader, optimizer, args, epoch)
            eval_it(model, valid_dataloader, args, epoch)


if __name__ == "__main__":
    args = get_args()
    main(args)
