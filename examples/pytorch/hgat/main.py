
# Main file for Minimum Implementation of hGATNet for node classification
# Import system necessaries
import argparse
# Pytorch related
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# dgl related
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from dgl.sampling import select_topk
from dgl.nn.pytorch.conv import GraphConv,GATConv
from dgl.data import CoraGraphDataset

from utils import Zeros


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
                 activation=None,
                 allow_zero_in_degree=False):
        super(HardGAO, self).__init__()
        self.num_heads = num_heads
        self.in_feats = in_feats
        self.out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.k = k
        # Initialize Parameters for Additive Attention
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
            subgraph.ndata['y'] = th.sigmoid(subgraph.ndata['y'])
            feat = th.matmul(th.diag(subgraph.ndata['y']),feat)
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
                 hid_dim,
                 out_dim,
                 residual=True,
                 head=1,
                 k=8,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2
                 ):
        super(HardGAM,self).__init__()
        self.in_dim = in_dim
        self.hid_dim= hid_dim
        self.out_dim= out_dim
        self.residual = residual
        self.feat_dropout = feat_drop
        self.attn_dropout = attn_drop
        self.head = head
        self.k = k
        self.negative_slope = negative_slope
        self.hgao = HardGAO(in_dim,hid_dim,
                            self.head,
                            k=self.k,
                            feat_drop=self.feat_dropout,
                            attn_drop=self.attn_dropout,
                            negative_slope=self.negative_slope
        )
        # As paper described the GCN should be weight free
        self.gcn = GraphConv(self.hid_dim,self.out_dim,weight=True,bias=False)
        # Implementation of residual as in Kaiming He ResNet
        if self.residual:
            if self.hid_dim==self.out_dim:
                self.res_m = Identity()
            else:
                self.res_m = nn.Linear(self.hid_dim,self.out_dim)
    
    def forward(self,g,n_feats):
        h = self.hgao(g,n_feats)
        h = h.view(h.shape[0],h.shape[1]*h.shape[2])
        ret = self.gcn(g,h)
        if self.residual:
            ret = ret+self.res_m(n_feats)
        return ret

class HardGANet(nn.Module):
    def __init__(self,
                 in_dim,
                 intm_dim=48,
                 hid_dim=16,
                 out_dim=7,
                 residual=True,
                 module_residual=True,
                 num_module=4,
                 k=8,
                 feat_dropout=0.,
                 attn_dropout=0.,
                 negative_slope=0.2
                 ):
        super(HardGANet,self).__init__()
        self.in_dim = in_dim
        self.intm_dim= intm_dim
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
        emb_gcn = GraphConv(in_dim,intm_dim)
        self.layers.append(emb_gcn)
        for n in range(self.num_module):
            if n==0:
                current_dim = intm_dim
            else:
                current_dim = hid_dim
            hgam = HardGAM(current_dim,
                           current_dim,
                           hid_dim,
                           residual=self.module_residual,
                           k=self.k,
                           feat_drop=self.feat_dropout,
                           attn_drop=self.attn_dropout,
                           negative_slope=self.negative_slope)
            self.layers.append(hgam)

        # Output GCN
        out_gcn = GraphConv(hid_dim,out_dim)
        self.layers.append(out_gcn)

        # Two different res module
        self.res = [Zeros(),Zeros()]
        if self.residual:
            self.res[0] = nn.Linear(intm_dim,hid_dim,bias=False)
            self.res[1] = Identity()
        
    def forward(self,g, n_feats):
        h = self.layers[0](g,n_feats)
        for n in range(self.num_module):
            h = self.layers[n+1](g,h) + self.res[min(1,n)](h)
        out = self.layers[-1](g,h)
        return out





# TODO: Implement Training Logic
def main(args):
    if args.dataset=='cora':
        dataset = CoraGraphDataset()
        graph = dataset[0]
    else:
        raise NotImplementedError

    if args.gpu >= 0 and th.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'

    num_classes = dataset.num_classes
    labels = graph.ndata.pop('label').to(device).long()


    n_feats = graph.ndata.pop('feat').to(device)
    in_dim  = n_feats.shape[-1]

    train_mask = graph.ndata.pop('train_mask')
    val_mask   = graph.ndata.pop('val_mask')
    test_mask  = graph.ndata.pop('test_mask')

    train_idx = th.nonzero(train_mask,as_tuple=False).squeeze().to(device)
    val_idx   = th.nonzero(val_mask,as_tuple=False).squeeze().to(device)
    test_idx  = th.nonzero(test_mask,as_tuple=False).squeeze().to(device)

    hgat_model = HardGANet(in_dim = in_dim,feat_dropout=0.12,attn_dropout=0.12,k=args.k)
    hgat_model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(hgat_model.parameters(),lr=args.lr,weight_decay=0.002)

    for epoch in range(args.max_epoch):
        hgat_model.train()
        logits = hgat_model.forward(graph,n_feats)
        tr_loss = loss_fn(logits[train_idx],labels[train_idx])
        tr_acc  = th.sum(th.argmax(logits[train_idx],1)==labels[train_idx]).item()/len(train_idx)

        valid_loss = loss_fn(logits[val_idx],labels[val_idx])
        valid_acc  = th.sum(th.argmax(logits[val_idx],1)==labels[val_idx]).item()/len(val_idx)

        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()

        print("In epoch {}, Train Acc: {:.4f} | Train Loss: {:4f}; Valid Acc: {:.4f} | Valid Loss: {:.4f}".
              format(epoch,tr_acc,tr_loss.item(),valid_acc,valid_loss.item()))

    hgat_model.eval()
    logits = hgat_model.forward(graph,n_feats)

    test_loss = loss_fn(logits[test_idx],labels[test_idx])
    test_acc  = th.sum(logits[test_idx].argmax(dim=1)==labels[test_idx]).item()/len(test_idx)
    print("Test Acc: {:.4f} | Test loss: {:.4f}".format(test_acc,test_loss.item()))

# TODO: Add Args parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MVP hGANet for Node Classification')
    parser.add_argument('--dataset',type=str,default="cora",help="DGL dataset for this MVP")
    parser.add_argument('--gpu',type=int,default=-1,help="GPU Index, Default: -1 Using CPU")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate. Default: 3e-1")
    parser.add_argument("--max_epoch", type=int, default=100, help="The max number of epoches. Default: 100")
    parser.add_argument("--num_module", type=int, default=4, help="The number of hGAO modules in GANet. Default: 4")
    parser.add_argument("--k",type=int,default=8, help="Number of k most important nodes for attention")

    args = parser.parse_args()
    print(args)
    
    main(args)