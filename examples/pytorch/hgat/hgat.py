
# Main file for Minimum Implementation of hGATNet for node classification
# Import system necessaries
import argparse
# Pytorch related
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
import time
import numpy as np

# dgl related
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from dgl.sampling import select_topk
from dgl.nn.pytorch.conv import GraphConv,GATConv
from dgl.data import CoraGraphDataset,CiteseerGraphDataset,PubmedGraphDataset
import dgl
from dgl.data import register_data_args

from utils import Zeros,accuracy, evaluate, EarlyStopping


# Layerwise implementation
class HardGAO(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 k=8,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 activation=F.elu,
                 allow_zero_in_degree=False):
        super(HardGAO, self).__init__()
        self.num_heads = num_heads
        self.in_feats = in_feats
        self.out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.k = k
        # Initialize Parameters for Additive Attention
        print('out feats:',self.out_feats)
        self.fc = nn.Linear(
            self.in_feats, self.out_feats * self.num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, self.num_heads, self.out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, self.num_heads, self.out_feats)))
        # Initialize Parameters for Hard Projection
        self.p = nn.Parameter(th.FloatTensor(size=(1,in_feats)))
        # Initialize Dropouts
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        # TODO: Maybe need exactly same initialization
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.p,gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def n2e_weight_transfer(self,edges):
        y = edges.src['y']
        return {'y':y}

    def forward(self, graph, feat, get_attention=False):
            # projection process to get importance vector y
            graph.ndata['y'] = th.abs(th.matmul(self.p,feat.T).view(-1))/th.norm(self.p,p=2)
            # Use edge message passing function to get the weight from src node
            graph.apply_edges(self.n2e_weight_transfer)
            # Select Top k neighbors
            subgraph = select_topk(graph,self.k,'y')
            # Sigmoid as information threshold
            subgraph.ndata['y'] = th.sigmoid(subgraph.ndata['y'])*2
            # Using vector matrix elementwise mul for acceleration
            feat = subgraph.ndata['y'].view(-1,1)*feat
            feat = self.feat_drop(feat)
            feat = self.fc(feat).view(-1, self.num_heads, self.out_feats)
            el = (feat * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat * self.attn_r).sum(dim=-1).unsqueeze(-1)
            # Assign the value on the subgraph
            subgraph.srcdata.update({'ft': feat, 'el': el})
            subgraph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            subgraph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(subgraph.edata.pop('e'))
            # compute softmax
            subgraph.edata['a'] = self.attn_drop(edge_softmax(subgraph, e))
            # message passing
            subgraph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = subgraph.dstdata['ft']
            
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, subgraph.edata['a']
            else:
                return rst

# TODO: Construct the hGAM
class HardGAM(nn.Module):
    # Stacking One hGAO and one GCN
    # Input-hidden; hidden-output
    # Residual means skeptical internal residual mentioned in paper
    def __init__(self,
                 in_dim,
                 out_dim,
                 head=1,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=True,
                 activation=F.elu,
                 k=8,
                 model='hgat'
                 ):
        super(HardGAM,self).__init__()
        self.in_dim = in_dim
        self.out_dim= out_dim
        self.residual = residual
        self.feat_dropout = feat_drop
        self.attn_dropout = attn_drop
        self.head = head
        self.k = k
        self.negative_slope = negative_slope
        if model == 'hgat':
            self.hgao = HardGAO(in_dim,out_dim,
                                self.head,
                                k=self.k,
                                feat_drop=self.feat_dropout,
                                attn_drop=self.attn_dropout,
                                negative_slope=self.negative_slope,
                                activation=activation
            )
        elif model == 'gat':
            self.hgao = GATConv(in_dim,out_dim,
                                self.head,
                                feat_drop=feat_drop,
                                attn_drop=attn_drop,
                                negative_slope=negative_slope,
                                activation=activation)
        else:
            raise NotImplementedError("No other model supported please use `hgao` or `gao`")
        # As paper described the GCN should be weight free
        self.gcn = GraphConv(self.out_dim*self.head,self.out_dim*self.head)
        # Implementation of skip connection
        if self.residual:
            if self.in_dim==self.out_dim:
                self.res_m = Identity()
            else:
                self.res_m = nn.Linear(self.in_dim,self.out_dim*head)
    
    def forward(self,g,n_feats):
        h = self.hgao(g,n_feats)
        hid = h.view(h.shape[0],h.shape[1]*h.shape[2])
        ret = self.gcn(g,hid).view(h.shape[0],-1,self.out_dim)
        #print(h.shape)
        ret = th.cat([ret,h],dim=2)
        return ret

class Modified_HardGAM(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 head=1,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=True,
                 activation=F.elu,
                 k=8,
                 model='hgat'
                 ):
        super(Modified_HardGAM,self).__init__()
        self.in_dim = in_dim
        self.out_dim= out_dim
        self.residual = residual
        self.feat_dropout = feat_drop
        self.attn_dropout = attn_drop
        self.head = head
        self.k = k
        self.negative_slope = negative_slope
        if model == 'hgat':
            self.hgao = HardGAO(in_dim,out_dim,
                                self.head,
                                k=self.k,
                                feat_drop=self.feat_dropout,
                                attn_drop=self.attn_dropout,
                                negative_slope=self.negative_slope,
                                activation=activation
            )
        elif model == 'gat':
            self.hgao = GATConv(in_dim,out_dim,
                                self.head,
                                feat_drop=feat_drop,
                                attn_drop=attn_drop,
                                negative_slope=negative_slope,
                                activation=activation)
        else:
            raise NotImplementedError("No other model supported please use `hgao` or `gao`")
        # As paper described the GCN should be weight free
        #self.gcn = GraphConv(self.out_dim*self.head,self.out_dim*self.head)
        # Implementation of skip connection
        if self.residual:
            if self.in_dim==self.out_dim:
                self.res_m = Identity()
            else:
                self.res_m = nn.Linear(self.in_dim,self.out_dim*head)
    
    def forward(self,g,n_feats):
        h = self.hgao(g,n_feats)
        if self.residual:
            ret = h + self.res_m(n_feats).view(h.shape[0],-1,h.shape[2])
        else:
            ret = h
        return ret

class HardGANet(nn.Module):
    def __init__(self,
                 g,
                 num_module,
                 in_dim,
                 hid_dim=8,
                 out_dim=7,
                 head=1,
                 activation=F.elu,
                 feat_dropout=0.,
                 attn_dropout=0.,
                 negative_slope=0.2,
                 residual=False,
                 model='hgat',
                 k=8,
                 module_residual=True
                 ):
        super(HardGANet,self).__init__()
        self.g = g
        self.in_dim = in_dim
        self.intm_dim= 48
        self.hid_dim= hid_dim
        self.out_dim= out_dim
        self.residual = residual
        self.module_residual = module_residual
        self.num_module = num_module
        self.k = 8
        self.feat_dropout = feat_dropout
        self.attn_dropout = attn_dropout
        self.negative_slope = negative_slope
        # Initialize layers
        self.layers = nn.ModuleList()
        # Use a layer of GCN as embedding
        emb_gcn = GraphConv(in_dim,
                           self.intm_dim,
                           activation=activation)
        self.layers.append(emb_gcn)
        cnt=0
        for n in range(self.num_module):
            if cnt==0:
                current_dim = self.intm_dim
            else:
                current_dim = current_dim + 2*hid_dim*head
            hgam = HardGAM(current_dim,
                           hid_dim,
                           head = head,
                           residual=self.module_residual,
                           k=self.k,
                           feat_drop=self.feat_dropout,
                           attn_drop=self.attn_dropout,
                           negative_slope=self.negative_slope,
                           model=model,
                           activation=activation)
            self.layers.append(hgam)
            cnt += 1

        # Output GCN
        current_dim = self.intm_dim if cnt==0 else current_dim + 2*hid_dim*head
        out_gcn = GraphConv(current_dim,
                           out_dim,
                           activation=None)
        self.layers.append(out_gcn)

        # Two different res module
        self.res = [Zeros(),Zeros()]
        if self.residual:
            self.res[0] = nn.Linear(self.intm_dim*head,hid_dim*head,bias=False)
            self.res[1] = Identity()
        
    def forward(self,n_feats):
        g = self.g
        h = self.layers[0](g,n_feats).flatten(1)
        for n in range(self.num_module):
            h = th.cat([self.layers[n+1](g,h).flatten(1),h],dim=1)
            #print(h.shape)
        out = self.layers[-1](g,h)
        return out

def main(args):
    # load and preprocess dataset
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    num_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    # create model
    heads = args.num_heads
    model = HardGANet(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual,
                args.model,
                args.k)
    print(model)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    if cuda:
        model.cuda()
    loss_fcn = th.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = th.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = accuracy(logits[train_mask], labels[train_mask])

        if args.fastmode:
            val_acc = accuracy(logits[val_mask], labels[val_mask])
        else:
            val_acc = evaluate(model, features, labels, val_mask)
            if args.early_stop:
                if stopper.step(val_acc, model):
                    break

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, n_edges / np.mean(dur) / 1000))

    print()
    if args.early_stop:
        model.load_state_dict(th.load('es_checkpoint.pt'))
    acc = evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=16,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    parser.add_argument('--model',type=str,default='hgat')
    parser.add_argument('--k',type=int,default=8)
    args = parser.parse_args()
    print(args)

    main(args)