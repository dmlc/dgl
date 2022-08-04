# change source code
import torchlayers as tl
def u_cat_e(edges):
    return {'m' : torch.hstack([edges.src['h'],edges.data['_edge_feats']])}


class EdgeSAGEConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 edge_feats=0,# add edge feats
                 edge_func = 'cat',
                 aggregator_type='mean',
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(EdgeSAGEConv,self).__init__()
        valid_aggre_types = {'mean', 'gcn', 'pool', 'lstm'}
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                'Invalid aggregator_type. Must be one of {}. '
                'But got {!r} instead.'.format(valid_aggre_types, aggregator_type)
            )

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        
        self.edge_feats = edge_feats
        
        
        #self._in_src_feats += edge_feats
        
          #self._in_dst_feats += self.edge_feats

        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
        if aggregator_type != 'gcn':
            self.fc_self = tl.Linear(out_feats, bias=False)
            
        
        self.fc_neigh = tl.Linear(out_feats, bias=False)
        
        
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer('bias', None)
            
        self.edge_func = edge_func
        self.fc_edge = None
        if self.edge_func =='cat':
            self.edge_func = u_cat_e
        
        if self.edge_func == 'transform':
            self.fc_edge = nn.Linear(edge_feats,out_feats)
        if self.edge_func =='as_edge_weights':
            self.fc_edge = nn.Linear(edge_feats,1)
            
        #self.reset_parameters()


    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        if self.fc_edge is not None:
            nn.init.xavier_uniform_(self.fc_edge.weight, gain=gain)
            
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)


    def _compatibility_check(self):
        """Address the backward compatibility issue brought by #2747"""
        if not hasattr(self, 'bias'):
            dgl_warning("You are loading a GraphSAGE model trained from a old version of DGL, "
                        "DGL automatically convert it to be compatible with latest version.")
            bias = self.fc_neigh.bias
            self.fc_neigh.bias = None
            if hasattr(self, 'fc_self'):
                if bias is not None:
                    bias = bias + self.fc_self.bias
                    self.fc_self.bias = None
            self.bias = bias

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox['m'] # (B, L, D)
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self._in_src_feats)),
             m.new_zeros((1, batch_size, self._in_src_feats)))
        _, (rst, _) = self.lstm(m, h)
        return {'neigh': rst.squeeze(0)}


    def forward(self, graph, feat, edge_weight=None,edge_feats=None):
        r"""

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N_{dst}, D_{out})`
            where :math:`N_{dst}` is the number of destination nodes in the input graph,
            :math:`D_{out}` is the size of the output feature.
        """
        self._compatibility_check()
        if isinstance(feat, tuple):
            feat[0] =feat[0]
            feat[1] = feat[1]
        else:
            feat = feat
        


        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            msg_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                msg_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            if edge_feats is not None:
                assert edge_feats.shape[0] == graph.number_of_edges()
                graph.edata['_edge_feats'] = edge_feats
                msg_fn = u_cat_e
                
            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            #lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            if self._aggre_type == 'mean':
                graph.srcdata['h'] = feat_src #self.fc_neigh(feat_src) if lin_before_mp else feat_src
                
                graph.update_all(msg_fn, fn.mean('m', 'neigh'))
                
                h_neigh = graph.dstdata['neigh']
#                 if not lin_before_mp:
#                     h_neigh = self.fc_neigh(h_neigh)
                h_neigh = self.fc_neigh(h_neigh)
                    
            elif self._aggre_type == 'gcn':
                check_eq_shape(feat)
                graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                if isinstance(feat, tuple):  # heterogeneous
                    graph.dstdata['h'] = self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                else:
                    if graph.is_block:
                        graph.dstdata['h'] = graph.srcdata['h'][:graph.num_dst_nodes()]
                    else:
                        graph.dstdata['h'] = graph.srcdata['h']
                graph.update_all(msg_fn, fn.sum('m', 'neigh'))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == 'pool':
                graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
                graph.update_all(msg_fn, fn.max('m', 'neigh'))
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])
            elif self._aggre_type == 'lstm':
                graph.srcdata['h'] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])
            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == 'gcn':
                rst = h_neigh
                
            else:
                rst = self.fc_self(h_self) + h_neigh

            # bias term
            if self.bias is not None:
                rst = rst + self.bias

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst


from dgl import nn as dglnn
from torch import nn

class EGRAPHSAGE(pl.LightningModule):
    def __init__(self, in_feats, n_hidden,edge_feats, n_classes ):
        super().__init__()
        self.save_hyperparameters()
        self.layers = nn.ModuleList()
        self.layers.append(EdgeSAGEConv(in_feats, n_hidden[0],edge_feats, 'cat','mean'))
        self.layers.append(EdgeSAGEConv(n_hidden[0], n_hidden[1],edge_feats, 'cat','mean'))

        #self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.n_classes = n_classes
        self.task_layer = nn.Linear(n_hidden[1],n_classes)
        self.activation = torch.sigmoid
        self.dropout = nn.Dropout(0.0)
        self.n_hidden = n_hidden
        
        self.train_auc = AUROC(pos_label=1)
        self.val_auc = AUROC(pos_label=1)
        self.automatic_optimization=True

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h,edge_feats=block.edata['feature'])
            h = torch.relu(h)
            # if l != len(self.layers) - 1:
            #     h = F.leaky_relu(h)
            #     h = self.dropout(h)
        h = self.dropout(self.task_layer(h))
        return self.activation(h)

    def training_step(self, batch, batch_idx):
        input_nodes, output_nodes, blocks = batch
        
        x = blocks[0].srcdata['feature']
        y = blocks[-1].dstdata['label']
    
        
        y_hat = self(blocks, x).flatten()
        
        loss = F.binary_cross_entropy(y_hat, y)
        
        self.train_auc(y_hat, y.long())
        
        self.log('train_auc', self.train_auc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_nodes, output_nodes, blocks = batch
        x = blocks[0].srcdata['feature']
        y = blocks[-1].dstdata['label']
        y_hat = self(blocks, x).flatten()
        self.val_auc(y_hat, y.long())
        
        self.log('val_auc', self.val_auc, prog_bar=True, on_step=True, on_epoch=True)

    def predict_step(self, batch,batch_idx): # use full back or use sampling but ensemble
        input_nodes, output_nodes, blocks = batch
        x = blocks[0].srcdata['feature']
        y_hat = self(blocks, x).flatten()
        return y_hat,blocks[-1].dstdata['nid']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.003, weight_decay=5e-7)
        return optimizer
