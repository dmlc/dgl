import dgl
import os
import torch as th
from dgl.data.utils import load_graphs, save_graphs
from dgl.sampling.neighbor import select_topk
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as Metric
import torch.nn.functional as F

def trans_feature(hg, het_gnn):
    for i in hg.ntypes:
        ndata = hg.nodes[i].data
        for j in ndata:
            het_gnn.nodes[i].data[j] = ndata[j]
    return het_gnn

def compute_loss(pos_score, neg_score):
    # an example hinge loss
    loss = []
    for i in pos_score:
        loss.append(F.logsigmoid(pos_score[i]))
        loss.append(F.logsigmoid(-neg_score[i]))
    loss = th.cat(loss)
    return -loss.mean()


def extract_feature(g, ntypes):
    input_features = {}
    for n in ntypes:
        ndata = g.srcnodes[n].data
        data = {}
        data['dw_embedding'] = ndata['dw_embedding']
        data['abstract'] = ndata['abstract']
        if n == 'paper':
            data['title'] = ndata['title']
            data['venue'] = ndata['venue']
            data['author'] = ndata['author']
            data['reference'] = ndata['reference']
        input_features[n] = data

    return input_features


def load_HIN(dataset):

    # which is used in HetGNN
    data_path = './academic.bin'
    category = 'author'
    num_classes = 4
    g = load_dgl_graph(data_path)
    g = g.long()
    return g, category, num_classes

def load_dgl_graph(path_file):
    g, _ = load_graphs(path_file)
    return g[0]

class hetgnn_graph():
    def __init__(self, hg, n_dataset):
        self.hg = hg
        self.g = dgl.to_homogeneous(hg).to('cpu')
        self.NID = self.g.ndata[dgl.NID]
        self.NTYPE = self.g.ndata[dgl.NTYPE]
        num_nodes = {}
        for i in range(th.max(self.NTYPE) + 1):
            num_nodes[self.hg.ntypes[i]] = int((self.NTYPE == i).sum())
        self.num_nodes = num_nodes
        self.weight_column = 'w'
        self.n_dataset = n_dataset

    def get_hetgnn_graph(self, length, walks, restart_prob):
        fname = './{}_het.bin'.format(
            self.n_dataset)
        if os.path.exists(fname):
            g, _ = load_graphs(fname)
            return g[0]
        else:
            g = self.build_hetgnn_graph(length, walks, restart_prob)
            save_graphs(fname, g)
            return g

    def build_hetgnn_graph(self, length, walks, restart_prob):
        #edges = [[[[],[]]] * len(self.num_nodes)] * len(self.num_nodes)
        edges = [[[[],[]], [[],[]], [[],[]]],
                 [[[],[]], [[],[]], [[],[]]],
                 [[[],[]], [[],[]], [[],[]]]]

        for i in range(self.g.number_of_nodes()):
            nodes = th.tensor([i]).repeat(walks)
            traces, types = dgl.sampling.random_walk(self.g, nodes, length=length, restart_prob=restart_prob)
            concat_vids, _, _, _ = dgl.sampling.pack_traces(traces, types)
            concat_types = th.index_select(self.NTYPE, 0, concat_vids)
            uid = concat_vids[0]
            utype = concat_types[0]
            for (vid, vtype) in zip(concat_vids, concat_types):
                edges[int(utype)][int(vtype)][0].append(self.NID[uid])
                edges[int(utype)][int(vtype)][1].append(self.NID[vid])

        edge_dict = {}
        k = {}
        num_ntypes = self.NTYPE.max() + 1
        for i in range(num_ntypes):
            for j in range(num_ntypes):
                edge = (self.hg.ntypes[j], self.hg.ntypes[j]+'-'+self.hg.ntypes[i], self.hg.ntypes[i])
                edge_dict[edge] = (th.tensor(edges[i][j][1]), th.tensor(edges[i][j][0]))
                if j == 2:
                    k[edge] = 3
                else:
                    k[edge] = 10

        neighbor_graph = dgl.heterograph(
            edge_dict,
            self.num_nodes
        )

        neighbor_graph = dgl.to_simple(neighbor_graph, return_counts=self.weight_column)
        counts = neighbor_graph.edata[self.weight_column]
        neighbor_graph = select_topk(neighbor_graph, k, self.weight_column)

        return neighbor_graph

def author_link_prediction(x, train_batch, test_batch):
    train_u, train_v, train_Y = train_batch
    test_u, test_v, test_Y = test_batch

    train_X = concat_u_v(x, th.tensor(train_u), th.tensor(train_v))
    test_X = concat_u_v(x, th.tensor(test_u), th.tensor(test_v))
    train_Y = th.tensor(train_Y)
    test_Y = th.tensor(test_Y)
    link_prediction(train_X, train_Y, test_X, test_Y)

def Hetgnn_evaluate(emd, labels, train_idx, test_idx):
    Y_train = labels[train_idx]
    Y_test = labels[test_idx]
    LR = LogisticRegression(max_iter=10000)
    X_train = emd[train_idx]
    X_test = emd[test_idx]
    LR.fit(X_train, Y_train)
    Y_pred = LR.predict(X_test)
    macro_f1, micro_f1 = f1_node_classification(Y_test, Y_pred)
    return micro_f1, macro_f1

''''''
def LR_pred(train_X, train_Y, test_X):
    LR = LogisticRegression(max_iter=10000)
    LR.fit(train_X, train_Y)
    pred_Y = LR.predict(test_X)
    #AUC_score = Metric.roc_auc_score(test_Y, pred_Y)
    return pred_Y

def link_prediction(train_X, train_Y, test_X, test_Y):
    pred_Y = LR_pred(train_X, train_Y, test_X)
    AUC_score = Metric.roc_auc_score(test_Y, pred_Y)
    print('AUC=%.4f' % AUC_score)
    macro_f1, micro_f1 = f1_node_classification(test_Y, pred_Y)
    print('macro_f1: {:.4f}, micro_f1: {:.4f}'.format(macro_f1, micro_f1))



''''''


def f1_node_classification(y_label, y_pred):
    macro_f1 = f1_score(y_label, y_pred, average='macro')
    micro_f1 = f1_score(y_label, y_pred, average='micro')
    return macro_f1, micro_f1

def load_link_pred(path_file):
    #path_file = './openhgnn/dataset/a_a_list_train.txt'
    u_list = []
    v_list = []
    label_list = []
    with open(path_file) as f:
        for i in f.readlines():
            u, v, label = i.strip().split(', ')
            u_list.append(int(u))
            v_list.append(int(v))
            label_list.append(int(label))
    return u_list, v_list, label_list

def concat_u_v(x, u_idx, v_idx):
    u = x[u_idx]
    v = x[v_idx]
    emd = th.cat((u, v), dim=1)
    return emd

