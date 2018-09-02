from torch.utils.data import Dataset as DatasetBase
import numpy as np
import torch.nn.functional as F
import torch

class Dataset(DatasetBase):
    def __init__(self, triplets):
        self.data = triplets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def negative_sampling(triplets, num_entity=0, negative_rate=0):
    pos_samples = np.array(triplets)
    size_of_batch = len(triplets)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

def perturb_and_get_rank(embedding, w, a, r, b, num_entity, batch_size=128):
    n_batch = (num_entity + batch_size - 1) // batch_size
    ranks = []
    for idx in range(n_batch):
        batch_start = idx * batch_size
        batch_end = min(num_entity, (idx + 1) * batch_size)
        batch_a = a[batch_start: batch_end]
        batch_r = r[batch_start: batch_end]
        emb_ar= (embedding[batch_a] * w[batch_r]).transpose(0, 1).unsqueeze(2) # size: D x E x 1
        target = b[batch_start: batch_end]
        emb_c = embedding.transpose(0, 1).unsqueeze(1) # size: D x 1 x V
        # out-prod and reduce sum
        out_prod = torch.bmm(emb_ar, emb_c) # size D x E x V
        score = torch.sum(out_prod, dim=0) # size E x V
        score = torch.sigmoid(score)
        ranks.append(sort_and_rank(score, target))
    return torch.cat(ranks)


# return MRR (raw, filtered), and Hits @ (1, 3, 10)
# default use CPU
def evaluate(model, test_triplets, num_entity, hits=[], eval_bz=128):
    with torch.no_grad():
        embedding, w = model.evaluate()
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]

        # perturb subject
        ranks_s = perturb_and_get_rank(embedding, w, o, r, s, num_entity, eval_bz)
        # perturb object
        ranks_o = perturb_and_get_rank(embedding, w, s, r, o, num_entity, eval_bz)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR: {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks < hit).float())
            print("Hits @ {}: {:.6f}".format(hit, avg_count.item()))



