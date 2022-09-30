import os
from time import time

import torch
import torch.optim as optim
from model import NGCF
from utility.batch_test import *
from utility.helper import early_stopping


def main(args):
    # Step 1: Prepare graph data and device ================================================================= #
    if args.gpu >= 0 and torch.cuda.is_available():
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    g = data_generator.g
    g = g.to(device)

    # Step 2: Create model and training components=========================================================== #
    model = NGCF(
        g, args.embed_size, args.layer_size, args.mess_dropout, args.regs[0]
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Step 3: training epoches ============================================================================== #
    n_batch = data_generator.n_train // args.batch_size + 1
    t0 = time()
    cur_best_pre_0, stopping_step = 0, 0
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0.0, 0.0, 0.0
        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(
                g, "user", "item", users, pos_items, neg_items
            )

            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(
                u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
            )
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = "Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]" % (
                    epoch,
                    time() - t1,
                    loss,
                    mf_loss,
                    emb_loss,
                )
                print(perf_str)
            continue  # end the current epoch and move to the next epoch, let the following evaluation run every 10 epoches

        # evaluate the model every 10 epoches
        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, g, users_to_test)
        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret["recall"])
        pre_loger.append(ret["precision"])
        ndcg_loger.append(ret["ndcg"])
        hit_loger.append(ret["hit_ratio"])

        if args.verbose > 0:
            perf_str = (
                "Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], "
                "precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]"
                % (
                    epoch,
                    t2 - t1,
                    t3 - t2,
                    loss,
                    mf_loss,
                    emb_loss,
                    ret["recall"][0],
                    ret["recall"][-1],
                    ret["precision"][0],
                    ret["precision"][-1],
                    ret["hit_ratio"][0],
                    ret["hit_ratio"][-1],
                    ret["ndcg"][0],
                    ret["ndcg"][-1],
                )
            )
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(
            ret["recall"][0],
            cur_best_pre_0,
            stopping_step,
            expected_order="acc",
            flag_step=5,
        )

        # early stop
        if should_stop == True:
            break

        if ret["recall"][0] == cur_best_pre_0 and args.save_flag == 1:
            torch.save(model.state_dict(), args.weights_path + args.model_name)
            print(
                "save the weights in path: ",
                args.weights_path + args.model_name,
            )

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = (
        "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]"
        % (
            idx,
            time() - t0,
            "\t".join(["%.5f" % r for r in recs[idx]]),
            "\t".join(["%.5f" % r for r in pres[idx]]),
            "\t".join(["%.5f" % r for r in hit[idx]]),
            "\t".join(["%.5f" % r for r in ndcgs[idx]]),
        )
    )
    print(final_perf)


if __name__ == "__main__":
    if not os.path.exists(args.weights_path):
        os.mkdir(args.weights_path)
    args.mess_dropout = eval(args.mess_dropout)
    args.layer_size = eval(args.layer_size)
    args.regs = eval(args.regs)
    print(args)
    main(args)
