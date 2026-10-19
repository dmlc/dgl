"""
utilily module for LightGCN.
Provide various functions.
"""

import numpy as np
import torch
import time
from dgl import graphbolt as gb

def RecallPrecision_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def NDCGatK_r(test_data,r,k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# Negative sampling function that ensures negative samples are not real edges.
# Based on the pattern: randomly select users, then sample positive and negative items for each user.
def sample_triplets(hetero_g, allPos, allPos_sets, n, etype=('user', 'interacts', 'item')):
    time_start = time.time()
    
    # 1. Get graph information
    num_users = hetero_g.num_nodes('user')
    num_items = hetero_g.num_nodes('item')
    device = hetero_g.device
    
    # 3. Randomly select n users (with replacement)
    selected_users = np.random.randint(0, num_users, n)
    
    # 4. Sample positive and negative items for each selected user
    users_list = []
    pos_items_list = []
    neg_items_list = []
    
    for user in selected_users:
        user = int(user)
        posForUser = allPos.get(user, [])

        if len(posForUser) == 0:
            continue
        # Sample one positive item randomly
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        
        # Sample negative item: ensure it's not in user's positive items
        pos_set = allPos_sets.get(user, set())
        max_retries = 100  # Prevent infinite loop
        negitem = None
        
        for _ in range(max_retries):
            negitem = np.random.randint(0, num_items)
            if negitem not in pos_set:
                break
        
        
        users_list.append(user)
        pos_items_list.append(positem)
        neg_items_list.append(negitem)
    
    # 5. Convert to tensors
    users = torch.tensor(users_list, dtype=torch.long, device=device)
    pos_items = torch.tensor(pos_items_list, dtype=torch.long, device=device)
    neg_items = torch.tensor(neg_items_list, dtype=torch.long, device=device)
    time_end = time.time()
    print("sample time cost:", time_end - time_start)
    return users, pos_items, neg_items

def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result