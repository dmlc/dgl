from dgl import backend as F
from dgl.data import RedditDataset
from dgl.data.utils import load_graphs, save_graphs

data = RedditDataset()
g = data.graph
g.ndata['features'] = F.tensor(data.features)
g.ndata['labels'] = F.tensor(data.labels)
g.ndata['test_mask'] = F.tensor(data.test_mask)
g.ndata['train_mask'] = F.tensor(data.train_mask)
g.ndata['val_mask'] = F.tensor(data.val_mask)
save_graphs('reddit/reddit.dgl', [g])
