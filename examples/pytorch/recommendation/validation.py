import torch
import tqdm
import numpy as np
from rec.utils import cuda

def compute_validation_rating(ml, h, b, model, test):
    n_users = len(ml.users)
    n_products = len(ml.products)

    h = h.cpu()
    b = b.cpu()
    M = h[:n_users] @ h[n_users:].t() + b[:n_users] + b[n_users:].t()
    field = 'valid' if not test else 'test'
    ratings = ml.ratings[ml.ratings[field]]
    avg_error = 0
    l = np.zeros(len(ratings))
    u_nids = [ml.user_ids_invmap[i] for i in ratings['user_id'].values]
    p_nids = [ml.product_ids_invmap[i] for i in ratings['product_id'].values]
    error = (ratings['rating'].values - M[u_nids, p_nids].numpy()) ** 2
    rmse = np.sqrt(error.mean())
    print(rmse)

    return error

def compute_validation_imp(ml, h, b, model, test):
    rr = []
    validation = not test
    n_users = len(ml.users)
    n_products = len(ml.products)

    h = h.cpu()
    b = b.cpu()
    M = h[:n_users] @ h[n_users:].t() + b[:n_users] + b[n_users:].t()

    with torch.no_grad():
        with tqdm.trange(n_users) as tq:
            for u_nid in tq:
                score = M[u_nid].clone()
                score[ml.p_train[u_nid]] = -10000
                score[ml.p_test[u_nid] if validation else ml.p_valid[u_nid]] = -10000
                rank = st.rankdata(-score)[ml.p_valid[u_nid] if validation else ml.p_test[u_nid]]
                rank = 1 / rank
                rr.append(rank.mean() if len(rank) > 0 else 0.)
                tq.set_postfix({'rank': rank.mean()})

    return np.array(rr)
