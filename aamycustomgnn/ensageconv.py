from dgl import function as fn

def u_cat_e(edges):
  return {'m' : torch.hstack([edges.src['h'],edges.data['w']])}


class ENSAGEConv(nn.Module): 
    """Graph convolution module used by the GraphSAGE model with edge features

    Parameters
    ----------
    in_feat : int
        Input feature size.
        
    out_feat : int
        Output feature size.
    """
    def __init__(self,out_feat):
        super(ENSAGEConv, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linear = tl.Linear(out_feat) #DenseResnet(in_feat * 2 + edge_feat,[in_feat * 2 + edge_feat]*2+[out_feat],0.0,False)
        #self.reset_parameters()

    def forward(self, g ,h=None):
        """Forward computation

        Parameters
        ----------
        g : Graph
            The input graph. a block or a subgraph
        h : Tensor
            The input node feature.
        w : Tensor
            The edge weight.
        """
        if h is None:
          h = g.srcdata['feature'] # 初始features
        else:
          h = h # 否则为上一层的节点的表征
        w = g.edata['feature']

        with g.local_scope():
            g.srcdata['h'] = h # not ndata
            g.edata['w'] = w

            g.update_all(message_func=u_cat_e, reduce_func=fn.mean('m', 'h_N')) # message_func的操作对象是edge，收集target node 对应的src node的features以及 src to dst的edges的features
            # reduce func的操作对象是nodes
            h_N = g.dstdata['h_N'] # 此时这个h n 是 所有的 target node的邻域的表征，shape 和 target node的node features 相同，那么关键问题就是这个target node的node features 怎么合并进来

            h = g.dstdata['feature']
            #h = h[:g.number_of_dst_nodes()] # 用这个方式来代替
            h_total = torch.cat([h, h_N], dim=1)
            return self.linear(h_total)

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
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)

        
