import torch
import torch.nn as nn
import torch.nn.functional as F

class EGTLayer(nn.Module):
    r"""EGTLayer for Edge-augmented Graph Transformer (EGT), as introduced in
    `Global Self-Attention as a Replacement for Graph Convolution
    Reference `<https://arxiv.org/pdf/2108.03348.pdf>`_

    Parameters
    ----------
    ndim : int
        Node embedding dimension.
    edim : int
        Edge embedding dimension.
    num_heads : int
        Number of attention heads, by which :attr: `ndim` is divisible.
    num_vns : int
        Number of virtual nodes.
    dropout : float, optional
        Dropout probability. Default: 0.0.
    attn_dropout : float, optional
        Attention dropout probability. Default: 0.0.
    activation : callable activation layer, optional
        Activation function. Default: nn.ELU().
    clip_logits_value : [float, float], optional
        Clamp attention weights to clip_logits_value. Default: [-5.0, 5.0].
    ffn_multiplier : float, optional
        Multiplier of the inner dimension in Feed Forward Network.
        Default: 2.0.
    scale_dot : bool, optional
        Whether to scale the attention weights. Default: True.
    scale_degree : bool, optional
        Whether to scale the degree encoding. Default: False.
    edge_update : bool, optional
        Whether to update the edge embedding. Default: True.

    Examples
    --------
    >>> import torch as th
    >>> from dgl.nn import EGTLayer

    >>> batch_size = 16
    >>> num_nodes = 100
    >>> ndim, edim = 128, 32
    >>> nfeat = th.rand(batch_size, num_nodes, ndim)
    >>> efeat = th.rand(batch_size, num_nodes, num_nodes, edim)
    >>> net = EGTLayer(
            ndim=ndim,
            edim=edim,
            num_heads=8,
            num_vns=4,
        )
    >>> out = net(nfeat, efeat)
    """

    def __init__(self,
                 ndim,
                 edim,
                 num_heads,
                 num_vns,
                 dropout=0,
                 attn_dropout=0,
                 activation=nn.ELU(),
                 clip_logits_value=[-5,5],
                 ffn_multiplier=2.,
                 scale_dot=True,
                 scale_degree=False,
                 edge_update=True,
                 ):
        super().__init__()
        self.ndim = ndim         
        self.edim = edim          
        self.num_heads = num_heads  
        self.num_vns = num_vns           
        self.dropout = dropout
        self.attn_dropout = attn_dropout       
        self.clip_logits_min = clip_logits_value[0]
        self.clip_logits_max = clip_logits_value[1]
        self.ffn_multiplier = ffn_multiplier
        self.scale_dot = scale_dot
        self.scale_degree = scale_degree      
        self.edge_update = edge_update        
        
        assert not (self.ndim % self.num_heads)
        self.dot_dim = self.ndim//self.num_heads
        
        self.mha_ln_h   = nn.LayerNorm(self.ndim)
        self.mha_ln_e   = nn.LayerNorm(self.edim)
        self.lin_E      = nn.Linear(self.edim, self.num_heads)
        self.lin_QKV    = nn.Linear(self.ndim, self.ndim*3)
        self.lin_G      = nn.Linear(self.edim, self.num_heads)
        
        self.ffn_fn     = activation
        self.lin_O_h    = nn.Linear(self.ndim, self.ndim)
        node_inner_dim  = round(self.ndim*self.ffn_multiplier)
        self.ffn_ln_h   = nn.LayerNorm(self.ndim)
        self.lin_W_h_1  = nn.Linear(self.ndim, node_inner_dim)
        self.lin_W_h_2  = nn.Linear(node_inner_dim, self.ndim)
        if self.dropout > 0:
            self.mha_drp_h  = nn.Dropout(self.dropout)
            self.ffn_drp_h  = nn.Dropout(self.dropout)
        
        if self.edge_update:
            self.lin_O_e    = nn.Linear(self.num_heads, self.edim)
            edge_inner_dim  = round(self.edim*self.ffn_multiplier)
            self.ffn_ln_e   = nn.LayerNorm(self.edim)
            self.lin_W_e_1  = nn.Linear(self.edim, edge_inner_dim)
            self.lin_W_e_2  = nn.Linear(edge_inner_dim, self.edim)
            if self.dropout > 0:
                self.mha_drp_e  = nn.Dropout(self.dropout)
                self.ffn_drp_e  = nn.Dropout(self.dropout)
    
    def forward(self, h, e, mask=None):
        """Forward computation.

        Parameters
        ----------
        h : torch.Tensor
            A 3D input tensor. Shape: (batch_size, N, :attr:`ndim`), where
            N is the maximum number of nodes.
        e : torch.Tensor
            Edge embedding used for attention computation and self update.
            Shape: (batch_size, N, N, :attr:`edim`).
        mask : torch.Tensor, optional
            The attention mask used for avoiding computation on invalid
            positions, where invalid positions are indicated by `-inf`.
            Shape: (batch_size, N, N, 1). Default: None.

        Returns
        -------
        h : torch.Tensor
            The output node embedding. Shape: (batch_size, N, :attr:`ndim`).
        e : torch.Tensor
            The output edge embedding. Shape: (batch_size, N, N, :attr:`edim`).
        """

        h_r1 = h
        e_r1 = e

        h_ln = self.mha_ln_h(h)
        e_ln = self.mha_ln_e(e)
        QKV = self.lin_QKV(h_ln)
        E = self.lin_E(e_ln)
        G = self.lin_G(e_ln)
        shp = QKV.shape
        Q, K, V = QKV.view(shp[0],shp[1],-1,self.num_heads).split(self.dot_dim,dim=2)
        A_hat = torch.einsum('bldh,bmdh->blmh', Q, K)
        if self.scale_dot:
            A_hat = A_hat * (self.dot_dim ** -0.5)
        H_hat = A_hat.clamp(self.clip_logits_min, self.clip_logits_max) + E

        if mask is None:
            gates = torch.sigmoid(G)
            A_tild = F.softmax(H_hat, dim=2) * gates
        else:
            gates = torch.sigmoid(G+mask)
            A_tild = F.softmax(H_hat+mask, dim=2) * gates

        if self.attn_dropout > 0:
            A_tild = F.dropout(A_tild, p=self.attn_dropout, training=self.training)

        V_att = torch.einsum('blmh,bmkh->blkh', A_tild, V)

        if self.scale_degree:
            degrees = torch.sum(gates,dim=2,keepdim=True)
            degree_scalers = torch.log(1+degrees)
            degree_scalers[:,:self.num_vns] = 1.
            V_att = V_att * degree_scalers

        V_att = V_att.reshape(shp[0],shp[1],self.num_heads*self.dot_dim)
        h = self.lin_O_h(V_att)

        if self.dropout > 0:
            h = self.mha_drp_h(h)
        h.add_(h_r1)
        h_r2 = h
        h_ln = self.ffn_ln_h(h)
        h = self.lin_W_h_2(self.ffn_fn(self.lin_W_h_1(h_ln)))
        if self.dropout > 0:
            h = self.ffn_drp_h(h)
        h.add_(h_r2)

        if self.edge_update:
            e = self.lin_O_e(H_hat)
            if self.dropout > 0:
                e = self.mha_drp_e(e)
            e.add_(e_r1)
            e_r2 = e
            e_ln = self.ffn_ln_e(e)
            e = self.lin_W_e_2(self.ffn_fn(self.lin_W_e_1(e_ln)))
            if self.dropout > 0:
                e = self.ffn_drp_e(e)
            e.add_(e_r2)

        return h, e
