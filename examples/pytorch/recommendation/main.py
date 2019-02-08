from model.pinsage import PinSage
from datasets.movielens import MovieLens

g, uid, mid = MovieLens('./ml-1m').todglgraph()

n_hidden = 100
n_layers = 3
batch_size = 32

# Use the prior graph to train on user-product pairs in the training set.
# Validate on validation set.
# Note that each user-product pair is counted twice, but I think it is OK
# since we can treat product negative sampling and user negative sampling
# ubiquitously.
g_prior_edges = g.filter_edges(lambda edges: edges.data['prior'])
g_train_edges = g.filter_edges(lambda edges: edges.data['train'])
g_valid_edges = g.filter_edges(lambda edges: edges.data['valid'])
g_prior = g.edge_subgraph(g_prior_edges)
g_prior_nid = g_prior.parent_nid

model = PinSage(g_prior, [n_hidden] * n_layers, 10, 5, 5)

for epoch in range(500):
    edge_batches = g_train_edges.split(batch_size)

    with tqdm.tqdm(edge_batches) as tq:
        for batch_id, batch in enumerate(tq):
            src, dst = g.find_edges(batch)
