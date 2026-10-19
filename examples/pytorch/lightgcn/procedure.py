"""
Procedure module for LightGCN.
Contains the train process and test process.
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
import time
import utils
import numpy as np
from config import args
from model import *

def train_lightgcn(dataset):
    g = dataset.graph
    device = torch.device(args.device)
    n_users = dataset.n_users
    n_items = dataset.n_items
    n_edges = dataset.n_edges
    model = LightGCN(n_users, n_items, args.recdim, args.layer).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        if epoch % 10 == 0 and epoch != 0:
            print(f"Test {epoch/10}")
            test_lightgcn(dataset, model)
        train_loader = dataset.build_train_dataset()
        model.train()
        total_loss = 0
        total_batch = n_edges // args.batch + 1
        # Epoch start
        epoch_start = time.time()

        for batch_idx, (batch_user, batch_pos, batch_neg) in enumerate(train_loader):
            batch_start = time.time()  
            # Batch start
            all_emb = model(g)
            user_emb = all_emb[batch_user]
            pos_emb = all_emb[batch_pos + n_users]
            neg_emb = all_emb[batch_neg + n_users]

            loss, reg_loss = model.bprLoss(user_emb, pos_emb, neg_emb, batch_user, batch_pos, batch_neg)
            batch_stage1 = time.time()
            reg_loss = reg_loss*args.decay
            loss=loss + reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Batch end
            batch_time = time.time()
            # print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, stage1 time: {batch_stage1 - batch_start:.4f}s, stage2 time: {batch_time - batch_stage1:.4f}s, Batch time: {batch_time - batch_start:.4f}s, Loss: {loss.item():.4f}")

        # Epoch end
        epoch_time = time.time() - epoch_start
        total_loss /= total_batch
        print(f"Epoch {epoch+1}/{args.epochs} finished. Epoch time: {epoch_time:.2f}s, Average BPR Loss: {total_loss:.4f}")

def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in args.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}

def test_lightgcn(dataset, model):       
    model = model.eval()
    max_K = max(args.topks)
    results = {'precision': np.zeros(len(args.topks)),
               'recall': np.zeros(len(args.topks)),
               'ndcg': np.zeros(len(args.topks))}
    test_dict = dataset.test_dict
    hetero_g = dataset.hetero_graph
    g = dataset.graph
    A = hetero_g.adj().csr()
    # test begin
    with torch.no_grad():
        users = list(test_dict.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // args.testbatch + 1
        test_dataset = TensorDataset(torch.Tensor(users))
        test_loader = DataLoader(test_dataset, batch_size=args.testbatch, shuffle=False)
        # batch begin
        for batch_user in test_loader:
            allPos = []
            batch_user_tensor = batch_user[0].long().to(dataset.device)
            for i in batch_user_tensor.tolist():
                i = int(i)
                start_col = A[0][i].item()
                end_col = A[0][i+1].item()
                allPos.append(A[1][start_col:end_col].tolist())
            groundTrue = [test_dict[int(u.item())] for u in batch_user_tensor]
            all_emb = model(g)
            user_emb = all_emb[batch_user_tensor]
            item_emb = all_emb[dataset.n_users:]
            rating = torch.sigmoid(torch.matmul(user_emb, item_emb.t()))
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_user_tensor.cpu())
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x))
        scale = float(args.testbatch/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        print(results)
