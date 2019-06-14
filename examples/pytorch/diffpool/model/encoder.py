import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np
from scipy.linalg import block_diag

import dgl

from .graphSage import GraphSage, GraphSageLayer
from tensorized_layers import *
import time


class GraphEncoder(nn.Module):
    """
    Baseline model that only leverages GraphSage to do graph readout
    """
    def __init__(self,input_dim, hidden_dim, embedding_dim, pred_hidden_dims,
                 label_dim, activation, n_layers, dropout, **kwargs):
        super(GraphEncoder, self).__init__()
        self.concat = kwargs['concat']
        self.bn = kwargs['bn']
        self.num_layers = n_layers
        self.num_aggs = 1
        self.bias = True
        if kwargs is not None:
            self.bias = kwargs['bias']

        self.act = activation
        self.label_dim = label_dim

        # control concatenation input for the prediction layer. (n-1) controls
        # for hidden dim * only the first n-1 layers (except for the last one,
        # which output embedding dimension
        if self.concat:
            self.pred_input_dim = hidden_dim * (n_layers -1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim # Without concatenation.
        # Build the MLP readout layer
        self.pred_model = self.build_pred_layers(self.pred_input_dim,
                                                 pred_hidden_dims, label_dim,
                                                 num_agg=self.num_aggs)

        self.gc_layers = nn.ModuleList()
        aggregator_type = kwargs['aggregator_type']
        #input layer
        self.gc_layers.append(GraphSageLayer(input_dim, hidden_dim, activation,
                                             dropout, aggregator_type))
        assert n_layers >= 3, "n_layers too few"
        for _ in range(n_layers - 2):
            self.gc_layers.append(GraphSageLayer(hidden_dim, hidden_dim,
                                                 activation, dropout,
                                                 aggregator_type))
        # output layer
        self.gc_layers.append(GraphSageLayer(hidden_dim, embedding_dim, None,
                                             dropout, aggregator_type))

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data,
                                                     gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def apply_bn(self, x):
        """
        Batch norm for 3D tensor X
        """
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def gcn_forward(self, g, h, gc_layers, cat=True):
        """
        Return gc_layer embedding cat.
        """
        block_readout = []
        for gc_layer in gc_layers[:-1]:
            h = gc_layer(g, h)
            if self.bn:
                h = self.apply_bn(h)
            block_readout.append(h)
        h = gc_layers[-1](g, h)
        block_readout.append(h)
        if cat:
            block = torch.cat(block_readout, dim=1) # N x F, F = F1 + F2 + ...
        else:
            block = h
        return block

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim,
                          num_agg=1):
        '''
        build prediction MLP layers
        '''
        # Num_agg denotes the aggregation mode
        pred_input_dim = pred_input_dim * num_agg
        if not pred_hidden_dims:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            pred_layers.append(nn.Linear(pred_input_dim,
                                         pred_hidden_dims[0], bias=True))

            if len(pred_hidden_dims) > 1:
                shifted_pred_dims = pred_hidden_dims[0:-1]
                pred_dims_pair = dict(zip(shifted_pred_dims, pred_hidden_dims[1:]))
                for (pred_dim_in, pred_dim_out) in pred_dims_pair.items():
                    pred_layers.append(nn.Linear(pred_dim_in, pred_dim_out, bias=True))

            pred_layers.append(nn.Linear(pred_hidden_dims[-1], label_dim,
                                         bias=True))
            pred_model = nn.Sequential(*pred_layers)

        return pred_model

    def forward(self, g):
        h = g.ndata['feat']
        out_all = []
        first_layer = self.gc_layers[0]
        h = first_layer(g, h)
        if self.bn:
            h = self.apply_bn(h)
        g.ndata['h'] = h
        out = dgl.max_nodes(g, 'h')
        out_all.append(out)
        for gc_layer in self.gc_layers[1:-1]:
            h = gc_layer(g, h)
            if self.bn:
                h = self.apply_bn(h)
            g.ndata['h'] = h
            out = dgl.max_nodes(g, 'h')
            out_all.append(out)
            if self.num_aggs == 2:
                out = dgl.sum_nodes(g, 'h')
                out_all.append(out)
        h = self.gc_layers[-1](g, h)
        g.ndata['h'] = h
        out = dgl.max_nodes(g, 'h')
        out_all.append(out)
        if self.num_aggs == 2:
            out = dgl.sum_nodes(g, 'h')
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)# This dimension should be G xC
        else:
            output = out
        ypred = self.pred_model(output)
        return ypred

    def loss(self, pred, label, loss_type='softmax'):
        '''
        loss function
        '''
        #softmax + CE
        if loss_type == 'softmax':
            criterion = nn.CrossEntropyLoss()
            loss = criterion(pred, label)
            return loss
        elif type == 'margin':
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size,
                                       self.label_dim).long().cuda()
            label_onehot.scatter_(1, label.view(-1, 1), 1)
            return torch.nn.MultiLabelmarginLoss()(pred, label_onehot)

class DiffPoolEncoder(GraphEncoder):
    """
    DiffPool
    """
    def __init__(self, input_dim, assign_input_dim, hidden_dim, embedding_dim, pred_hidden_dims,
                 assign_hidden_dim, label_dim, activation, n_layers,
                 assign_n_layers, dropout, n_pooling, linkpred, **kwargs):
        super(DiffPoolEncoder, self).__init__(input_dim, hidden_dim,
                                              embedding_dim, pred_hidden_dims,
                                              label_dim, activation, n_layers,
                                              dropout, **kwargs)


        self.linkpred = linkpred
        self.n_pooling = n_pooling
        self.batch_size = kwargs['batch_size']
        self.link_pred_loss = []
        self.entropy_loss = []
        self.hloss = HLoss()

        self.gc_layers_after_pool = nn.ModuleList()

        for _ in range(n_pooling):
            gc_layer_after_pool = nn.ModuleList()
            aggregator_type = kwargs['aggregator_type']
            gc_layer_after_pool.append(GraphSageLayer(self.pred_input_dim,
                                                      hidden_dim, activation,
                                                      dropout,
                                                      aggregator_type))
            for _ in range(n_layers - 2):
                gc_layer_after_pool.append(GraphSageLayer(hidden_dim, hidden_dim,
                                                          activation, dropout,
                                                          aggregator_type))
            gc_layer_after_pool.append(GraphSageLayer(hidden_dim,
                                                      embedding_dim, None,
                                                      dropout,
                                                      aggregator_type))

        self.gc_layers_after_pool.append(gc_layer_after_pool)


        # assignment module based on a seperate GC
        assign_dims = []
        if assign_n_layers == -1:
            # default case
            assign_n_layers = n_layers

        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.gc_layers_assign = nn.ModuleList()
        self.pred_layers_assign = nn.ModuleList()

        assign_dim = kwargs['assign_dim'] # initialize assign_dim
        for _ in range(n_pooling):
            assign_dims.append(assign_dim)
            gc_layer_assign = nn.ModuleList()
            gc_layer_assign.append(GraphSageLayer(assign_input_dim,
                                                  assign_hidden_dim,
                                                  activation, dropout,
                                                  aggregator_type))
            for _ in range(assign_n_layers - 2):
                gc_layer_assign.append(GraphSageLayer(assign_hidden_dim,
                                                      assign_hidden_dim,
                                                      activation, dropout,
                                                      aggregator_type))
            gc_layer_assign.append(GraphSageLayer(assign_hidden_dim,
                                                  assign_dim, activation,
                                                  dropout, aggregator_type))


            assign_pred_input_dim =\
            assign_hidden_dim * (n_layers - 1) + assign_dim if kwargs['concat'] else assign_dim
            pred_layer_assign = self.build_pred_layers(assign_pred_input_dim,
                                                       [], assign_dim,
                                                       num_agg=1)
            self.gc_layers_assign.append(gc_layer_assign)
            self.pred_layers_assign.append(pred_layer_assign)

            # next pooling layer
            assign_input_dim = embedding_dim
            assign_dim = int(assign_dim / kwargs['batch_size'] *
                             kwargs['pool_ratio']) * kwargs['batch_size']


        # final prediction layer
        self.pred_model = self.build_pred_layers(self.pred_input_dim *
                                                 (n_pooling + 1),
                                                 pred_hidden_dims, label_dim,
                                                 num_agg=self.num_aggs)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data,
                                                     gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, g):
        self.link_pred_loss = []
        self.entropy_loss = []
        h = g.ndata['feat']
        # node feature for assignment matrix computation is the same as the
        # original node feature
        h_a = h

        out_all = []

        # we use GCN blocks to get an embedding first
        total_time = []
        comment = []
        start = time.time()
        g_embedding = self.gcn_forward(g, h, self.gc_layers)
        end = time.time() - start
        total_time.append(end)
        comment.append("pre_assignment graph convolution")

        g.ndata['h'] = g_embedding

        out = dgl.max_nodes(g, 'h')
        out_all.append(out)
        if self.num_aggs == 2:
            out = dgl.sum_nodes(g, 'h')
            out_all.append(out)

        for i in range(self.n_pooling):
            # register as member because we need to use it in loss function
            start = time.time()
            self.assign_tensor = self.gcn_forward(g, h_a,
                                                  self.gc_layers_assign[i],
                                                  False)
            # so that each node will not be assigned to clusters of other
            # graphs
            assign_tensor_masks = []
            for g_n_nodes in g.batch_num_nodes:
                mask =torch.ones((g_n_nodes,
                                  int(self.assign_tensor.size()[1]/self.batch_size)))
                assign_tensor_masks.append(mask)
            mask = torch.FloatTensor(block_diag(*assign_tensor_masks)).cuda()

            end = time.time() - start
            total_time.append(end)
            comment.append("assignment matrix calculation")


            start = time.time()
            self.assign_tensor = masked_softmax(self.assign_tensor, mask,
                                                memory_efficient=False)
            end = time.time() - start
            total_time.append(end)
            comment.append("mask softmax calculation")

            if not self.training:
                _, idx = torch.max(self.assign_tensor, dim=1)
                row_idx = list(range(self.assign_tensor.size()[0]))
                sparse_assign = torch.zeros_like(self.assign_tensor)
                sparse_assign[row_idx, list(idx)] = 1
                self.assign_tensor = sparse_assign

            # X_n = S^T X_{n-1}
            start = time.time()
            h = torch.matmul(torch.t(self.assign_tensor),g_embedding)
            end = time.time() - start
            total_time.append(end)
            comment.append("lifting node feature matrix")

            # we need to take out the adjacency matrix to compute S^TAS
            # current device defaulted to cuda
            device = torch.device('cuda:0')
            adj = g.adjacency_matrix(ctx=device)
            node_seg_list = g.batch_num_nodes
            # S^TAS
            start = time.time()
            adj_new = torch.sparse.mm(adj, self.assign_tensor)
            adj_new = torch.mm(torch.t(self.assign_tensor), adj_new)
            end = time.time() - start
            total_time.append(end)
            comment.append("calculating next graph adj matrix")

            h_a = h# updating assignment input and embedding input

            pooled_g_n_nodes = self.assign_tensor.size()[1]

            start = time.time()
            pooled_g = dgl.DGLGraph()
            # create the new pooled graph (batch version)
            pooled_g.add_nodes(pooled_g_n_nodes)
            adj_indices = adj_new.to_sparse()._indices()
            # add edges
            pooled_g.add_edges(adj_indices[1], adj_indices[0])
            end = time.time() - start
            total_time.append(end)
            comment.append("consrtructing new graph")

            # edge weight is not set on pooled graph yet.\TODO

            # at this point we have block diagonally dense graph (a batch
            # graph)
            start = time.time()
            embedding_tensor = self.gcn_forward(pooled_g, h, self.gc_layers_after_pool[i])
            end = time.time() - start
            total_time.append(end)
            comment.append("after_assignment graph convolution")

            pooled_g.ndata['h'] = embedding_tensor
            out_temp = []
            for index in node_seg_list:
                sum_slice = torch.sum(pooled_g.ndata['h'][:index,:], dim=0,
                                      keepdim=True)
                out_temp.append(sum_slice)
            out = torch.cat(out_temp, dim=0)

            out_all.append(out)
            if self.num_aggs == 2:
                out_temp = []
                for index in node_seg_list:
                    sum_slice = torch.sum(pooled_g.ndata['h'][:index,:],dim=1)
                    out_temp.append(sum_slice)
                out_all.append(out)

            # L_{lp} = A - S^TS
            start = time.time()
            current_lp_loss = torch.norm(adj.to_dense() - torch.mm(self.assign_tensor,
                                                                   torch.t(self.assign_tensor))) / np.power(g.number_of_nodes(),2)
            self.link_pred_loss.append(current_lp_loss)

            # entropy loss
            current_entropy_loss =\
            self.hloss(self.assign_tensor, mask) / np.power(g.number_of_nodes(), 2)
            self.entropy_loss.append(current_entropy_loss)
            end = time.time() - start
            comment.append("calculating loss")
            total_time.append(end)

            g = pooled_g

        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        start = time.time()
        ypred = self.pred_model(output)
        end = time.time() - start
        total_time.append(end)
        total_time = np.array(total_time)
        print((total_time/np.sum(total_time))*100)
        print(comment)
        return ypred

    def loss(self, pred, label):
        original_loss = super(DiffPoolEncoder, self).loss(pred, label)
        # default entropy loss enabled
        entropy_loss = sum(self.entropy_loss)
        original_loss = original_loss + entropy_loss
        if self.linkpred:
            self.link_loss = sum(self.link_pred_loss)

            return original_loss + self.link_loss
        else:
            return original_loss

def masked_softmax(matrix, mask, dim=-1, memory_efficient=True,
                   mask_fill_value=-1e32):
    '''
    masked_softmax for dgl batch graph
    '''
    if mask is None:
        result = torch.nn.functional.softmax(matrix, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < matrix.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            result = torch.nn.functional.softmax(matrix * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_matrix = matrix.masked_fill((1-mask).byte(),
                                               mask_fill_value)
            result = torch.nn.functional.softmax(masked_matrix, dim=dim)
    return result

def masked_log_softmax(matrix, mask, dim=-1):
    '''
    masked log softmax for dgl batch graph
    '''
    if mask is not None:
        mask = mask.float()
        while mask.dim() < matrix.dim():
            mask = mask.unsqueeze(1)
        matrix = matrix + (mask + 1e-32).log()
    return torch.nn.functional.log_softmax(matrix, dim=dim)

class HLoss(nn.Module):
    '''
    entropy loss
    '''
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x, mask):
        hl = masked_softmax(x,mask, memory_efficient=False)*masked_log_softmax(x,mask)
        hl = hl.sum()
        return -hl
